// image.cpp - Implementation avec auto-détection SD
#include "image.h"
#include "esphome/core/log.h"
#include "esphome/core/application.h"
#include <sys/stat.h>
#include <dirent.h>
#include <cstring>
#include <algorithm>

#ifdef USE_ESP32
#include "esp_vfs.h"
#include "esp_vfs_fat.h"
#include "driver/sdmmc_host.h"
#include "sdmmc_cmd.h"
#endif

// Inclusion conditionnelle pour JPEG/PNG si disponibles
#ifdef USE_IMAGE_DECODE_JPEG
#include <jpeglib.h>
#include <setjmp.h>
#endif

#ifdef USE_IMAGE_DECODE_PNG
#include <png.h>
#endif

namespace esphome {
namespace image {

static const char *const TAG = "image";

// Static member definition
SDFileReader Image::global_sd_reader_ = nullptr;

Image::Image(const uint8_t *data_start, int width, int height, ImageType type, Transparency transparency)
    : width_(width), height_(height), type_(type), data_start_(data_start), transparency_(transparency) {
  switch (type) {
    case IMAGE_TYPE_BINARY:
      this->bpp_ = 1;
      break;
    case IMAGE_TYPE_GRAYSCALE:
      this->bpp_ = transparency == TRANSPARENCY_ALPHA_CHANNEL ? 16 : 8;
      break;
    case IMAGE_TYPE_RGB565:
      this->bpp_ = transparency == TRANSPARENCY_ALPHA_CHANNEL ? 24 : 16;
      break;
    case IMAGE_TYPE_RGB:
      this->bpp_ = transparency == TRANSPARENCY_ALPHA_CHANNEL ? 32 : 24;
      break;
  }
}

int Image::get_width() const { return this->width_; }
int Image::get_height() const { return this->height_; }
ImageType Image::get_type() const { return this->type_; }

// ===== AUTO-DETECTION SD FUNCTIONS =====

bool file_exists(const std::string &path) {
  struct stat st;
  return (stat(path.c_str(), &st) == 0) && S_ISREG(st.st_mode);
}

bool directory_exists(const std::string &path) {
  struct stat st;
  return (stat(path.c_str(), &st) == 0) && S_ISDIR(st.st_mode);
}

std::vector<std::string> find_sd_mount_points() {
  std::vector<std::string> mount_points;
  
  // Points de montage courants pour cartes SD sur ESP32
  const std::vector<std::string> common_mounts = {
    "/sdcard",
    "/sd", 
    "/mnt/sd",
    "/mnt/sdcard",
    "/media/sd",
    "/mnt",
    "/"  // Racine du système de fichiers
  };
  
  for (const auto &mount : common_mounts) {
    if (directory_exists(mount)) {
      mount_points.push_back(mount);
      ESP_LOGD(TAG, "Found potential SD mount point: %s", mount.c_str());
    }
  }
  
  return mount_points;
}

std::string auto_detect_sd_root() {
  auto mount_points = find_sd_mount_points();
  
  // Teste chaque point de montage
  for (const auto &mount : mount_points) {
    DIR *dir = opendir(mount.c_str());
    if (dir != nullptr) {
      // Vérifie s'il y a des fichiers dans ce répertoire
      struct dirent *entry;
      bool has_files = false;
      int file_count = 0;
      
      while ((entry = readdir(dir)) != nullptr && file_count < 10) {
        // Skip les entrées . et ..
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
          continue;
        }
        
        file_count++;
        if (entry->d_type == DT_REG) {  // Fichier régulier
          has_files = true;
        }
      }
      closedir(dir);
      
      ESP_LOGD(TAG, "Mount point %s: %d entries, has_files=%d", 
               mount.c_str(), file_count, has_files);
      
      if (file_count > 0) {  // Le répertoire contient quelque chose
        ESP_LOGI(TAG, "Auto-detected SD root: %s (%d entries)", mount.c_str(), file_count);
        return mount;
      }
    }
  }
  
  ESP_LOGW(TAG, "Could not auto-detect SD root, using fallback: /sdcard");
  return "/sdcard";
}

std::vector<std::string> generate_search_paths(const std::string &original_path, const std::string &sd_root) {
  std::vector<std::string> search_paths;
  
  // Normalise le chemin original
  std::string normalized = original_path;
  std::replace(normalized.begin(), normalized.end(), '\\', '/');
  
  // Supprime les doubles slashes
  size_t pos = 0;
  while ((pos = normalized.find("//", pos)) != std::string::npos) {
    normalized.replace(pos, 2, "/");
  }
  
  // 1. Chemin exact tel que configuré
  search_paths.push_back(normalized);
  
  // 2. Avec la racine auto-détectée
  if (normalized.front() == '/') {
    // Chemin absolu - combine avec racine
    if (sd_root != "/") {
      search_paths.push_back(sd_root + normalized);
    }
  } else {
    // Chemin relatif
    search_paths.push_back(sd_root + "/" + normalized);
  }
  
  // 3. Variations courantes
  std::string filename = normalized.substr(normalized.find_last_of("/") + 1);
  
  // Chemins avec différentes racines
  const std::vector<std::string> alt_roots = {"/sdcard", "/sd", "/mnt/sd", "/"};
  for (const auto &root : alt_roots) {
    if (root != sd_root) {
      search_paths.push_back(root + "/" + filename);
      if (normalized.find("/") != std::string::npos) {
        // Garde la structure de répertoires
        std::string subpath = normalized;
        if (subpath.front() == '/') subpath = subpath.substr(1);
        search_paths.push_back(root + "/" + subpath);
      }
    }
  }
  
  // 4. Dans le répertoire racine direct
  search_paths.push_back("/" + filename);
  search_paths.push_back(filename);
  
  // Supprime les doublons
  std::sort(search_paths.begin(), search_paths.end());
  search_paths.erase(std::unique(search_paths.begin(), search_paths.end()), search_paths.end());
  
  return search_paths;
}

// ===== IMAGE LOADING FUNCTIONS =====

bool Image::mount_sd_card() {
  if (this->sdcard_mounted_) {
    return true;  // Déjà monté
  }

#ifdef USE_ESP32
  // Configuration basique du montage SD
  sdmmc_host_t host = SDMMC_HOST_DEFAULT();
  sdmmc_slot_config_t slot_config = SDMMC_SLOT_CONFIG_DEFAULT();
  
  esp_vfs_fat_sdmmc_mount_config_t mount_config = {
    .format_if_mount_failed = false,
    .max_files = 5,
    .allocation_unit_size = 16 * 1024
  };

  sdmmc_card_t *card;
  const char* mount_point = "/sdcard";
  
  esp_err_t ret = esp_vfs_fat_sdmmc_mount(mount_point, &host, &slot_config, 
                                          &mount_config, &card);
  
  if (ret == ESP_OK) {
    ESP_LOGI(TAG, "SD card mounted successfully at %s", mount_point);
    this->sdcard_mounted_ = true;
    return true;
  } else {
    ESP_LOGD(TAG, "SD card mount failed: %s (may already be mounted)", esp_err_to_name(ret));
    // Teste si déjà monté
    if (directory_exists("/sdcard")) {
      ESP_LOGI(TAG, "SD card appears to be already mounted");
      this->sdcard_mounted_ = true;
      return true;
    }
  }
#endif
  
  return false;
}

bool Image::read_sd_file(const std::string &path, std::vector<uint8_t> &data) {
  ESP_LOGI(TAG, "Attempting to read SD file: %s", path.c_str());
  
  // 1. Utilise le lecteur de fichier spécifique s'il existe
  if (this->sd_file_reader_ && this->sd_file_reader_(path, data)) {
    ESP_LOGI(TAG, "File read successfully using specific reader: %zu bytes", data.size());
    return true;
  }
  
  // 2. Utilise le lecteur global s'il existe
  if (global_sd_reader_ && global_sd_reader_(path, data)) {
    ESP_LOGI(TAG, "File read successfully using global reader: %zu bytes", data.size());
    return true;
  }
  
  // 3. Auto-détection et lecture directe
  ESP_LOGI(TAG, "No SD file reader available - starting auto-detection");
  
  // Auto-détecte la racine SD
  std::string sd_root = auto_detect_sd_root();
  ESP_LOGI(TAG, "Auto-detected SD root: %s", sd_root.c_str());
  
  // Génère les chemins de recherche
  auto search_paths = generate_search_paths(path, sd_root);
  
  ESP_LOGI(TAG, "Generated %zu search paths for: %s", search_paths.size(), path.c_str());
  for (size_t i = 0; i < search_paths.size(); i++) {
    ESP_LOGD(TAG, "  %zu. %s", i + 1, search_paths[i].c_str());
  }
  
  // Teste chaque chemin
  for (const auto &search_path : search_paths) {
    ESP_LOGD(TAG, "Trying path: %s", search_path.c_str());
    
    if (file_exists(search_path)) {
      ESP_LOGI(TAG, "✓ Found file at: %s", search_path.c_str());
      
      FILE *file = fopen(search_path.c_str(), "rb");
      if (!file) {
        ESP_LOGE(TAG, "Cannot open file: %s (errno: %d - %s)", 
                 search_path.c_str(), errno, strerror(errno));
        continue;
      }
      
      // Obtient la taille du fichier
      fseek(file, 0, SEEK_END);
      long file_size = ftell(file);
      fseek(file, 0, SEEK_SET);
      
      if (file_size <= 0 || file_size > 10 * 1024 * 1024) {  // Max 10MB
        ESP_LOGE(TAG, "Invalid file size: %ld bytes", file_size);
        fclose(file);
        continue;
      }
      
      // Lit le fichier
      data.resize(file_size);
      size_t bytes_read = fread(data.data(), 1, file_size, file);
      fclose(file);
      
      if (bytes_read == static_cast<size_t>(file_size)) {
        ESP_LOGI(TAG, "✓ File read successfully: %zu bytes from %s", bytes_read, search_path.c_str());
        return true;
      } else {
        ESP_LOGE(TAG, "Failed to read complete file: %zu/%ld bytes", bytes_read, file_size);
        data.clear();
        continue;
      }
    } else {
      ESP_LOGD(TAG, "✗ File not found: %s", search_path.c_str());
    }
  }
  
  ESP_LOGE(TAG, "Failed to read SD file: %s (tried %zu paths)", path.c_str(), search_paths.size());
  return false;
}

size_t Image::get_expected_buffer_size() const {
  size_t pixel_size = 0;
  
  switch (this->type_) {
    case IMAGE_TYPE_BINARY:
      return ((this->width_ + 7) / 8) * this->height_;
    
    case IMAGE_TYPE_GRAYSCALE:
      pixel_size = (this->transparency_ == TRANSPARENCY_ALPHA_CHANNEL) ? 2 : 1;
      break;
    
    case IMAGE_TYPE_RGB565:
      pixel_size = (this->transparency_ == TRANSPARENCY_ALPHA_CHANNEL) ? 3 : 2;
      break;
    
    case IMAGE_TYPE_RGB:
      pixel_size = (this->transparency_ == TRANSPARENCY_ALPHA_CHANNEL) ? 4 : 3;
      break;
  }
  
  return this->width_ * this->height_ * pixel_size;
}

bool Image::load_from_sd() {
  if (this->sd_path_.empty()) {
    ESP_LOGE(TAG, "No SD path configured");
    return false;
  }
  
  ESP_LOGI(TAG, "Loading image from SD: %s", this->sd_path_.c_str());
  
  // Assure-toi que la SD est montée
  if (!mount_sd_card()) {
    ESP_LOGW(TAG, "SD card not mounted, but trying file access anyway");
  }
  
  std::vector<uint8_t> file_data;
  if (!read_sd_file(this->sd_path_, file_data)) {
    ESP_LOGE(TAG, "Failed to read SD file: %s", this->sd_path_.c_str());
    return false;
  }
  
  ESP_LOGI(TAG, "Successfully read %zu bytes from SD", file_data.size());
  
  // Décode l'image selon son format
  if (decode_image_from_sd()) {
    return true;
  }
  
  // Si le décodage échoue, essaie le décodage des formats d'image
  bool decode_success = false;
  
  // Détecte le format d'image par les premiers bytes
  if (file_data.size() >= 4) {
    // JPEG
    if (file_data[0] == 0xFF && file_data[1] == 0xD8) {
      ESP_LOGI(TAG, "Detected JPEG format");
      decode_success = decode_jpeg_data(file_data);
    }
    // PNG
    else if (file_data.size() >= 8 && 
             file_data[0] == 0x89 && file_data[1] == 0x50 && 
             file_data[2] == 0x4E && file_data[3] == 0x47) {
      ESP_LOGI(TAG, "Detected PNG format");
      decode_success = decode_png_data(file_data);
    }
    // Raw data
    else {
      ESP_LOGI(TAG, "Assuming raw image data format");
      size_t expected_size = get_expected_buffer_size();
      if (file_data.size() >= expected_size) {
        this->sd_buffer_ = std::move(file_data);
        decode_success = true;
        ESP_LOGI(TAG, "Raw data loaded successfully");
      } else {
        ESP_LOGE(TAG, "File size mismatch: got %zu bytes, expected %zu", 
                 file_data.size(), expected_size);
      }
    }
  }
  
  if (decode_success) {
    ESP_LOGI(TAG, "SD image loaded and decoded successfully");
    return true;
  } else {
    ESP_LOGE(TAG, "Failed to decode SD image data");
    return false;
  }
}

bool Image::decode_image_from_sd() {
  // Placeholder - implémentation dépend du format d'image spécifique
  // Pour l'instant, on assume que c'est du raw data
  return false;  // Force l'utilisation des décodeurs spécialisés
}

bool Image::decode_jpeg_data(const std::vector<uint8_t> &jpeg_data) {
#ifdef USE_IMAGE_DECODE_JPEG
  ESP_LOGI(TAG, "Decoding JPEG data (%zu bytes)", jpeg_data.size());
  
  struct jpeg_decompress_struct cinfo;
  struct jpeg_error_mgr jerr;
  
  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_decompress(&cinfo);
  
  jpeg_mem_src(&cinfo, jpeg_data.data(), jpeg_data.size());
  
  if (jpeg_read_header(&cinfo, TRUE) != JPEG_HEADER_OK) {
    ESP_LOGE(TAG, "Failed to read JPEG header");
    jpeg_destroy_decompress(&cinfo);
    return false;
  }
  
  // Configuration du décodage selon le type d'image cible
  switch (this->type_) {
    case IMAGE_TYPE_GRAYSCALE:
      cinfo.out_color_space = JCS_GRAYSCALE;
      break;
    case IMAGE_TYPE_RGB:
    case IMAGE_TYPE_RGB565:
      cinfo.out_color_space = JCS_RGB;
      break;
    case IMAGE_TYPE_BINARY:
      cinfo.out_color_space = JCS_GRAYSCALE;
      break;
  }
  
  jpeg_start_decompress(&cinfo);
  
  // Alloue le buffer de sortie
  size_t buffer_size = get_expected_buffer_size();
  this->sd_buffer_.resize(buffer_size);
  
  // Décode ligne par ligne
  JSAMPARRAY buffer = (*cinfo.mem->alloc_sarray)
    ((j_common_ptr) &cinfo, JPOOL_IMAGE, cinfo.output_width * cinfo.output_components, 1);
  
  size_t output_pos = 0;
  while (cinfo.output_scanline < cinfo.output_height) {
    jpeg_read_scanlines(&cinfo, buffer, 1);
    
    // Convertit la ligne selon le format de sortie
    for (unsigned int x = 0; x < cinfo.output_width && output_pos < buffer_size; x++) {
      switch (this->type_) {
        case IMAGE_TYPE_GRAYSCALE:
          this->sd_buffer_[output_pos++] = buffer[0][x];
          break;
        case IMAGE_TYPE_RGB:
          if (x * 3 + 2 < cinfo.output_width * cinfo.output_components) {
            this->sd_buffer_[output_pos++] = buffer[0][x * 3];     // R
            this->sd_buffer_[output_pos++] = buffer[0][x * 3 + 1]; // G
            this->sd_buffer_[output_pos++] = buffer[0][x * 3 + 2]; // B
          }
          break;
        case IMAGE_TYPE_RGB565:
          if (x * 3 + 2 < cinfo.output_width * cinfo.output_components) {
            uint8_t r = buffer[0][x * 3] >> 3;
            uint8_t g = buffer[0][x * 3 + 1] >> 2;
            uint8_t b = buffer[0][x * 3 + 2] >> 3;
            uint16_t rgb565 = (r << 11) | (g << 5) | b;
            this->sd_buffer_[output_pos++] = rgb565 >> 8;
            this->sd_buffer_[output_pos++] = rgb565 & 0xFF;
          }
          break;
        case IMAGE_TYPE_BINARY:
          this->sd_buffer_[output_pos++] = (buffer[0][x] > 128) ? 0xFF : 0x00;
          break;
      }
    }
  }
  
  jpeg_finish_decompress(&cinfo);
  jpeg_destroy_decompress(&cinfo);
  
  ESP_LOGI(TAG, "JPEG decoded successfully, output size: %zu bytes", output_pos);
  return true;
  
#else
  ESP_LOGE(TAG, "JPEG support not compiled in");
  return false;
#endif
}

bool Image::decode_png_data(const std::vector<uint8_t> &png_data) {
#ifdef USE_IMAGE_DECODE_PNG
  ESP_LOGI(TAG, "Decoding PNG data (%zu bytes)", png_data.size());
  
  png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
  if (!png_ptr) {
    ESP_LOGE(TAG, "Failed to create PNG read struct");
    return false;
  }
  
  png_infop info_ptr = png_create_info_struct(png_ptr);
  if (!info_ptr) {
    ESP_LOGE(TAG, "Failed to create PNG info struct");
    png_destroy_read_struct(&png_ptr, nullptr, nullptr);
    return false;
  }
  
  // Configuration de la source de données
  struct PngDataSource {
    const uint8_t* data;
    size_t size;
    size_t pos;
  } source = {png_data.data(), png_data.size(), 0};
  
  png_set_read_fn(png_ptr, &source, [](png_structp png_ptr, png_bytep data, png_size_t length) {
    PngDataSource* src = static_cast<PngDataSource*>(png_get_io_ptr(png_ptr));
    if (src->pos + length > src->size) {
      png_error(png_ptr, "Read beyond PNG data");
    }
    memcpy(data, src->data + src->pos, length);
    src->pos += length;
  });
  
  png_read_info(png_ptr, info_ptr);
  
  int width = png_get_image_width(png_ptr, info_ptr);
  int height = png_get_image_height(png_ptr, info_ptr);
  int color_type = png_get_color_type(png_ptr, info_ptr);
  int bit_depth = png_get_bit_depth(png_ptr, info_ptr);
  
  ESP_LOGI(TAG, "PNG info: %dx%d, color_type=%d, bit_depth=%d", width, height, color_type, bit_depth);
  
  // Transformations selon le format de sortie
  if (bit_depth == 16) {
    png_set_strip_16(png_ptr);
  }
  
  if (color_type == PNG_COLOR_TYPE_PALETTE) {
    png_set_palette_to_rgb(png_ptr);
  }
  
  if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8) {
    png_set_expand_gray_1_2_4_to_8(png_ptr);
  }
  
  png_read_update_info(png_ptr, info_ptr);
  
  // Alloue le buffer
  size_t buffer_size = get_expected_buffer_size();
  this->sd_buffer_.resize(buffer_size);
  
  // Lit l'image ligne par ligne
  png_bytep row = static_cast<png_bytep>(malloc(png_get_rowbytes(png_ptr, info_ptr)));
  size_t output_pos = 0;
  
  for (int y = 0; y < height && output_pos < buffer_size; y++) {
    png_read_row(png_ptr, row, nullptr);
    
    // Convertit selon le format de sortie
    for (int x = 0; x < width && output_pos < buffer_size; x++) {
      switch (this->type_) {
        case IMAGE_TYPE_GRAYSCALE:
          if (color_type == PNG_COLOR_TYPE_GRAY) {
            this->sd_buffer_[output_pos++] = row[x];
          } else if (color_type == PNG_COLOR_TYPE_RGB) {
            // Convertit RGB en grayscale
            uint8_t gray = (row[x * 3] * 299 + row[x * 3 + 1] * 587 + row[x * 3 + 2] * 114) / 1000;
            this->sd_buffer_[output_pos++] = gray;
          }
          break;
        case IMAGE_TYPE_RGB:
          if (color_type == PNG_COLOR_TYPE_RGB) {
            this->sd_buffer_[output_pos++] = row[x * 3];
            this->sd_buffer_[output_pos++] = row[x * 3 + 1];
            this->sd_buffer_[output_pos++] = row[x * 3 + 2];
          }
          break;
        // Ajouter d'autres formats selon les besoins
      }
    }
  }
  
  free(row);
  png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
  
  ESP_LOGI(TAG, "PNG decoded successfully, output size: %zu bytes", output_pos);
  return true;
  
#else
  ESP_LOGE(TAG, "PNG support not compiled in");
  return false;
#endif
}

// ===== PIXEL ACCESS FUNCTIONS =====

uint8_t Image::get_data_byte_(size_t pos) const {
  if (this->sd_runtime_ && !this->sd_buffer_.empty()) {
    if (pos < this->sd_buffer_.size()) {
      return this->sd_buffer_[pos];
    } else {
      ESP_LOGW(TAG, "SD buffer access out of bounds: %zu >= %zu", pos, this->sd_buffer_.size());
      return 0;
    }
  }
  return this->data_start_[pos];
}

Color Image::get_pixel(int x, int y, Color color_on, Color color_off) const {
  if (x < 0 || y < 0 || x >= this->width_ || y >= this->height_) {
    return color_off;
  }
  
  switch (this->type_) {
    case IMAGE_TYPE_BINARY:
      return this->get_binary_pixel_(x, y) ? color_on : color_off;
    case IMAGE_TYPE_GRAYSCALE:
      return this->get_grayscale_pixel_(x, y);
    case IMAGE_TYPE_RGB:
      return this->get_rgb_pixel_(x, y);
    case IMAGE_TYPE_RGB565:
      return this->get_rgb565_pixel_(x, y);
    default:
      return color_off;
  }
}

bool Image::get_binary_pixel_(int x, int y) const {
  const size_t width_8 = (this->width_ + 7u) / 8u;
  const size_t pos = x + y * width_8 * 8u;
  return this->get_data_byte_(pos / 8u) & (0x80 >> (pos % 8u));
}

Color Image::get_grayscale_pixel_(int x, int y) const {
  const size_t pos = x + y * this->width_;
  const uint8_t gray = this->get_data_byte_(pos);
  return Color(gray, gray, gray);
}

Color Image::get_rgb_pixel_(int x, int y) const {
  const size_t pos = (x + y * this->width_) * 3;
  const uint8_t r = this->get_data_byte_(pos + 0);
  const uint8_t g = this->get_data_byte_(pos + 1);
  const uint8_t b = this->get_data_byte_(pos + 2);
  return Color(r, g, b);
}

Color Image::get_rgb565_pixel_(int x, int y) const {
  const size_t pos = (x + y * this->width_) * 2;
  const uint8_t hi = this->get_data_byte_(pos + 0);
  const uint8_t lo = this->get_data_byte_(pos + 1);
  const uint16_t rgb565 = (hi << 8) | lo;
  
  const uint8_t r = (rgb565 >> 11) << 3;
  const uint8_t g = ((rgb565 >> 5) & 0x3F) << 2;
  const uint8_t b = (rgb565 & 0x1F) << 3;
  
  return Color(r, g, b);
}

void Image::draw(int x, int y, display::Display *display, Color color_on, Color color_off) {
  // Charge l'image depuis la SD si nécessaire
  if (this->sd_runtime_ && this->sd_buffer_.empty()) {
    ESP_LOGD(TAG, "Loading SD image for display: %s", this->sd_path_.c_str());
    if (!this->load_from_sd()) {
      ESP_LOGE(TAG, "Failed to load SD image for display: %s", this->sd_path_.c_str());
      return;
    }
  }
  
  // Dessine l'image pixel par pixel
  for (int img_x = 0; img_x < this->width_; img_x++) {
    for (int img_y = 0; img_y < this->height_; img_y++) {
      Color pixel_color = this->get_pixel(img_x, img_y, color_on, color_off);
      if (this->transparency_ == TRANSPARENCY_CHROMA_KEY && pixel_color == color_off) {
        continue;  // Skip transparent pixels
      }
      display->draw_pixel_at(x + img_x, y + img_y, pixel_color);
    }
  }
}

#ifdef USE_LVGL
lv_img_dsc_t *Image::get_lv_img_dsc() {
  // Charge depuis la SD si nécessaire
  if (this->sd_runtime_ && this->sd_buffer_.empty()) {
    ESP_LOGD(TAG, "Loading SD image for LVGL: %s", this->sd_path_.c_str());
    if (!this->load_from_sd()) {
      ESP_LOGE(TAG, "Failed to load SD image for LVGL: %s", this->sd_path_.c_str());
      return nullptr;
    }
  }

  this->dsc_.header.always_zero = 0;
  this->dsc_.header.w = this->width_;
  this->dsc_.header.h = this->height_;
  this->dsc_.data_size = this->sd_buffer_.empty() ? 
    ((this->width_ * this->bpp_ + 7) / 8) * this->height_ : 
    this->sd_buffer_.size();
  this->dsc_.header.cf = LV_IMG_CF_RAW;

  switch (this->type_) {
    case IMAGE_TYPE_BINARY:
      this->dsc_.header.cf = LV_IMG_CF_ALPHA_1BIT;
      break;
    case IMAGE_TYPE_GRAYSCALE:
      if (this->transparency_ == TRANSPARENCY_ALPHA_CHANNEL) {
        this->dsc_.header.cf = LV_IMG_CF_ALPHA_8BIT;
      } else {
        this->dsc_.header.cf = LV_IMG_CF_RAW;
      }
      break;
    case IMAGE_TYPE_RGB565:
      if (this->transparency_ == TRANSPARENCY_ALPHA_CHANNEL) {
        this->dsc_.header.cf = LV_IMG_CF_RGB565A8;
      } else {
        this->dsc_.header.cf = LV_IMG_CF_RGB565;
      }
      break;
    case IMAGE_TYPE_RGB:
      if (this->transparency_ == TRANSPARENCY_ALPHA_CHANNEL) {
        this->dsc_.header.cf = LV_IMG_CF_RGB888;
      } else {
        this->dsc_.header.cf = LV_IMG_CF_RGB888;
      }
      break;
  }

  // Utilise les données SD si disponibles, sinon les données intégrées
  if (!this->sd_buffer_.empty()) {
    this->dsc_.data = this->sd_buffer_.data();
    ESP_LOGD(TAG, "LVGL image using SD data: %zu bytes", this->sd_buffer_.size());
  } else {
    this->dsc_.data = this->data_start_;
    ESP_LOGD(TAG, "LVGL image using embedded data");
  }

  return &this->dsc_;
}
#endif

}  // namespace image
}  // namespace esphome
