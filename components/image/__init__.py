from __future__ import annotations

import hashlib
import io
import logging
from pathlib import Path
import re

from PIL import Image, UnidentifiedImageError

from esphome import core, external_files
import esphome.codegen as cg
from esphome.components.const import CONF_BYTE_ORDER
import esphome.config_validation as cv
from esphome.const import (
    CONF_DEFAULTS,
    CONF_DITHER,
    CONF_FILE,
    CONF_ICON,
    CONF_ID,
    CONF_PATH,
    CONF_RAW_DATA_ID,
    CONF_RESIZE,
    CONF_SOURCE,
    CONF_TYPE,
    CONF_URL,
)
from esphome.core import CORE, HexInt

_LOGGER = logging.getLogger(__name__)

DOMAIN = "image"
# NOTE: sd_mmc_card dépendance conditionnelle - sera ajoutée automatiquement si nécessaire
DEPENDENCIES = ["display"]

# Configuration pour la liaison avec sd_mmc_card
CONF_SD_MMC_CARD_ID = "sd_mmc_card_id"

image_ns = cg.esphome_ns.namespace("image")
ImageType = image_ns.enum("ImageType")

# Classes spécialisées pour images SD (pas de stockage en flash)
SDImage_ = image_ns.class_("SDImage", image_ns.class_("Image"))  # Hérite de Image mais avec comportement SD

CONF_OPAQUE = "opaque"
CONF_CHROMA_KEY = "chroma_key"
CONF_ALPHA_CHANNEL = "alpha_channel"
CONF_INVERT_ALPHA = "invert_alpha"
CONF_IMAGES = "images"

TRANSPARENCY_TYPES = (
    CONF_OPAQUE,
    CONF_CHROMA_KEY,
    CONF_ALPHA_CHANNEL,
)

TransparencyType = image_ns.enum("TransparencyType")
CONF_TRANSPARENCY = "transparency"

# If the MDI file cannot be downloaded within this time, abort.
IMAGE_DOWNLOAD_TIMEOUT = 30  # seconds

SOURCE_LOCAL = "local"
SOURCE_WEB = "web"
SOURCE_SD_CARD = "sd_card"

SOURCE_MDI = "mdi"
SOURCE_MDIL = "mdil"
SOURCE_MEMORY = "memory"

MDI_SOURCES = {
    SOURCE_MDI: "https://raw.githubusercontent.com/Templarian/MaterialDesign/master/svg/",
    SOURCE_MDIL: "https://raw.githubusercontent.com/Pictogrammers/MaterialDesignLight/refs/heads/master/svg/",
    SOURCE_MEMORY: "https://raw.githubusercontent.com/Pictogrammers/Memory/refs/heads/main/src/svg/",
}

Image_ = image_ns.class_("Image")
INSTANCE_TYPE = Image_


def get_image_type_enum(type):
    return getattr(ImageType, f"IMAGE_TYPE_{type.upper()}")


def get_transparency_enum(transparency):
    return getattr(TransparencyType, f"TRANSPARENCY_{transparency.upper()}")


class ImageEncoder:
    """
    Superclass of image type encoders
    """

    # Control which transparency options are available for a given type
    allow_config = {CONF_ALPHA_CHANNEL, CONF_CHROMA_KEY, CONF_OPAQUE}

    # All imageencoder types are valid
    @staticmethod
    def validate(value):
        return value

    def __init__(self, width, height, transparency, dither, invert_alpha):
        """
        :param width:  The image width in pixels
        :param height:  The image height in pixels
        :param transparency: Transparency type
        :param dither: Dither method
        :param invert_alpha: True if the alpha channel should be inverted; for monochrome formats inverts the colours.
        """
        self.transparency = transparency
        self.width = width
        self.height = height
        self.data = [0 for _ in range(width * height)]
        self.dither = dither
        self.index = 0
        self.invert_alpha = invert_alpha
        self.path = ""

    def convert(self, image, path):
        """
        Convert the image format
        :param image:  Input image
        :param path:  Path to the image file
        :return: converted image
        """
        return image

    def encode(self, pixel):
        """
        Encode a single pixel
        """

    def end_row(self):
        """
        Marks the end of a pixel row
        :return:
        """


def is_alpha_only(image: Image):
    """
    Check if an image (assumed to be RGBA) is only alpha
    """
    # Any alpha data?
    if image.split()[-1].getextrema()[0] == 0xFF:
        return False
    return all(b.getextrema()[1] == 0 for b in image.split()[:-1])


class ImageBinary(ImageEncoder):
    allow_config = {CONF_OPAQUE, CONF_INVERT_ALPHA, CONF_CHROMA_KEY}

    def __init__(self, width, height, transparency, dither, invert_alpha):
        self.width8 = (width + 7) // 8
        super().__init__(self.width8, height, transparency, dither, invert_alpha)
        self.bitno = 0

    def convert(self, image, path):
        if is_alpha_only(image):
            image = image.split()[-1]
        return image.convert("1", dither=self.dither)

    def encode(self, pixel):
        if self.invert_alpha:
            pixel = not pixel
        if pixel:
            self.data[self.index] |= 0x80 >> (self.bitno % 8)
        self.bitno += 1
        if self.bitno == 8:
            self.bitno = 0
            self.index += 1

    def end_row(self):
        """
        Pad rows to a byte boundary
        """
        if self.bitno != 0:
            self.bitno = 0
            self.index += 1


class ImageGrayscale(ImageEncoder):
    allow_config = {CONF_ALPHA_CHANNEL, CONF_CHROMA_KEY, CONF_INVERT_ALPHA, CONF_OPAQUE}

    def convert(self, image, path):
        if is_alpha_only(image):
            if self.transparency != CONF_ALPHA_CHANNEL:
                _LOGGER.warning(
                    "Grayscale image %s is alpha only, but transparency is set to %s",
                    path,
                    self.transparency,
                )
                self.transparency = CONF_ALPHA_CHANNEL
            image = image.split()[-1]
        return image.convert("LA")

    def encode(self, pixel):
        b, a = pixel
        if self.transparency == CONF_CHROMA_KEY:
            if b == 1:
                b = 0
            if a != 0xFF:
                b = 1
        if self.invert_alpha:
            b ^= 0xFF
        if self.transparency == CONF_ALPHA_CHANNEL:
            if a != 0xFF:
                b = a
        self.data[self.index] = b
        self.index += 1


class ImageRGB565(ImageEncoder):
    def __init__(self, width, height, transparency, dither, invert_alpha):
        stride = 3 if transparency == CONF_ALPHA_CHANNEL else 2
        super().__init__(
            width * stride,
            height,
            transparency,
            dither,
            invert_alpha,
        )
        self.big_endian = True

    def set_big_endian(self, big_endian: bool) -> None:
        self.big_endian = big_endian

    def convert(self, image, path):
        return image.convert("RGBA")

    def encode(self, pixel):
        r, g, b, a = pixel
        r = r >> 3
        g = g >> 2
        b = b >> 3
        if self.transparency == CONF_CHROMA_KEY:
            if r == 0 and g == 1 and b == 0:
                g = 0
            elif a < 128:
                r = 0
                g = 1
                b = 0
        rgb = (r << 11) | (g << 5) | b
        if self.big_endian:
            self.data[self.index] = rgb >> 8
            self.index += 1
            self.data[self.index] = rgb & 0xFF
            self.index += 1
        else:
            self.data[self.index] = rgb & 0xFF
            self.index += 1
            self.data[self.index] = rgb >> 8
            self.index += 1
        if self.transparency == CONF_ALPHA_CHANNEL:
            if self.invert_alpha:
                a ^= 0xFF
            self.data[self.index] = a
            self.index += 1


class ImageRGB(ImageEncoder):
    def __init__(self, width, height, transparency, dither, invert_alpha):
        stride = 4 if transparency == CONF_ALPHA_CHANNEL else 3
        super().__init__(
            width * stride,
            height,
            transparency,
            dither,
            invert_alpha,
        )

    def convert(self, image, path):
        return image.convert("RGBA")

    def encode(self, pixel):
        r, g, b, a = pixel
        if self.transparency == CONF_CHROMA_KEY:
            if r == 0 and g == 1 and b == 0:
                g = 0
            elif a < 128:
                r = 0
                g = 1
                b = 0
        self.data[self.index] = r
        self.index += 1
        self.data[self.index] = g
        self.index += 1
        self.data[self.index] = b
        self.index += 1
        if self.transparency == CONF_ALPHA_CHANNEL:
            if self.invert_alpha:
                a ^= 0xFF
            self.data[self.index] = a
            self.index += 1


class ReplaceWith:
    """
    Placeholder class to provide feedback on deprecated features
    """

    allow_config = {CONF_ALPHA_CHANNEL, CONF_CHROMA_KEY, CONF_OPAQUE}

    def __init__(self, replace_with):
        self.replace_with = replace_with

    def validate(self, value):
        raise cv.Invalid(
            f"Image type {value} is removed; replace with {self.replace_with}"
        )


IMAGE_TYPE = {
    "BINARY": ImageBinary,
    "GRAYSCALE": ImageGrayscale,
    "RGB565": ImageRGB565,
    "RGB": ImageRGB,
    "TRANSPARENT_BINARY": ReplaceWith("'type: BINARY' and 'transparency: chroma_key'"),
    "RGB24": ReplaceWith("'type: RGB'"),
    "RGBA": ReplaceWith("'type: RGB' and 'transparency: alpha_channel'"),
}


def compute_local_image_path(value) -> Path:
    url = value[CONF_URL] if isinstance(value, dict) else value
    h = hashlib.new("sha256")
    h.update(url.encode())
    key = h.hexdigest()[:8]
    base_dir = external_files.compute_local_file_dir(DOMAIN)
    return base_dir / key


def local_path(value):
    value = value[CONF_PATH] if isinstance(value, dict) else value
    return str(CORE.relative_config_path(value))


def sd_card_path(value):
    """Retourne le chemin complet sur la racine de la SD card"""
    value = value[CONF_PATH] if isinstance(value, dict) else value
    # Supprime un éventuel slash en début pour éviter les doublons
    value = value.lstrip("/\\")
    full_path = "/" + value  # chemin à partir de la racine de la SD
    _LOGGER.info(f"Chemin SD résolu: {full_path}")
    return full_path


def is_sd_card_path(path_str: str) -> bool:
    """Check if a path is an SD card path"""
    if not isinstance(path_str, str):
        return False
    path_str = path_str.strip()
    return (
        path_str.startswith("sd_card/") or 
        path_str.startswith("sd_card//") or
        path_str.startswith("/sdcard/") or
        path_str.startswith("sdcard/") or
        path_str.startswith("//") or
        path_str.startswith("/sd/") or
        path_str.startswith("sd/")
    )


def download_file(url, path):
    external_files.download_content(url, path, IMAGE_DOWNLOAD_TIMEOUT)
    return str(path)


def download_gh_svg(value, source):
    mdi_id = value[CONF_ICON] if isinstance(value, dict) else value
    base_dir = external_files.compute_local_file_dir(DOMAIN) / source
    path = base_dir / f"{mdi_id}.svg"

    url = MDI_SOURCES[source] + mdi_id + ".svg"
    return download_file(url, path)


def download_image(value):
    value = value[CONF_URL] if isinstance(value, dict) else value
    return download_file(value, compute_local_image_path(value))


def is_svg_file(file):
    if not file:
        return False
    # Pour les fichiers SD card, on ne peut pas vérifier le contenu
    if isinstance(file, str) and is_sd_card_path(file):
        return file.lower().endswith('.svg')
    with open(file, "rb") as f:
        return "<svg" in str(f.read(1024))


def validate_cairosvg_installed():
    try:
        import cairosvg
    except ImportError as err:
        raise cv.Invalid(
            "Please install the cairosvg python package to use this feature. "
            "(pip install cairosvg)"
        ) from err

    major, minor, _ = cairosvg.__version__.split(".")
    if major < "2" or major == "2" and minor < "2":
        raise cv.Invalid(
            "Please update your cairosvg installation to at least 2.2.0. "
            "(pip install -U cairosvg)"
        )


def validate_file_shorthand(value):
    value = cv.string_strict(value)
    
    # Vérification pour les chemins SD card - VERSION CORRIGÉE
    if is_sd_card_path(value):
        _LOGGER.info(f"SD card image detected: {value}")
        return value  # Retourne le chemin tel quel pour SD card
    
    parts = value.strip().split(":")
    if len(parts) == 2 and parts[0] in MDI_SOURCES:
        match = re.match(r"^[a-zA-Z0-9\-]+$", parts[1])
        if match is None:
            raise cv.Invalid(f"Could not parse mdi icon name from '{value}'.")
        return download_gh_svg(parts[1], parts[0])

    if value.startswith("http://") or value.startswith("https://"):
        return download_image(value)

    value = cv.file_(value)
    return local_path(value)


def normalize_to_sd_path(path: str) -> str:
    """
    Normalise le chemin vers un format unifié pour la carte SD.
    Cette fonction génère maintenant les métadonnées pour la recherche automatique.
    """
    p = str(path).strip()
    p = p.replace("\\", "/")
    # collapse multiple slashes
    p = re.sub(r"/+", "/", p)
    
    # Nettoie et normalise le chemin
    if p.startswith("/sd_card/"):
        rest = p[9:]  # Enlever "/sd_card/"
        normalized = "/" + rest
    elif p.startswith("sd_card/"):
        rest = p[8:]  # Enlever "sd_card/"
        normalized = "/" + rest
    elif p.startswith("/sdcard/"):
        rest = p[8:]  # Enlever "/sdcard/"
        normalized = "/" + rest
    elif p.startswith("sdcard/"):
        rest = p[7:]  # Enlever "sdcard/"
        normalized = "/" + rest
    elif not p.startswith("/"):
        normalized = "/" + p
    else:
        normalized = p
    
    _LOGGER.info(f"Chemin SD normalisé: {path} -> {normalized}")
    return normalized


def generate_sd_search_paths(original_path: str) -> list:
    """
    Génère une liste complète de chemins de recherche possibles pour la carte SD.
    Cette fonction sera utilisée par le runtime C++ pour trouver automatiquement
    la racine de la carte SD en testant différents points de montage.
    """
    normalized = normalize_to_sd_path(original_path)
    filename = Path(normalized).name
    parent_dirs = Path(normalized).parent.parts[1:]  # Skip root '/'
    
    search_paths = []
    
    # Chemin normalisé exact
    search_paths.append(normalized)
    
    # Points de montage standards pour cartes SD (ordre de priorité)
    mount_points = [
        "/sdcard",        # Android/ESP32 standard
        "/sd",            # ESP32 classique
        "/mnt/sd",        # Linux mount point
        "/mnt/sdcard",    # Alternative Linux
        "/media/sd",      # Ubuntu/Debian auto-mount
        "/mnt",           # Point de montage générique
        "/",              # Racine système (dernier recours)
    ]
    
    for mount_point in mount_points:
        if parent_dirs:
            # Avec sous-répertoires
            full_path = f"{mount_point}/" + "/".join(parent_dirs) + f"/{filename}"
            search_paths.append(full_path)
        else:
            # Directement à la racine du point de montage
            search_paths.append(f"{mount_point}/{filename}")
    
    # Chemins relatifs (sans slash initial)
    if parent_dirs:
        search_paths.append("/".join(parent_dirs) + f"/{filename}")
    search_paths.append(filename)
    
    # Supprime les doublons tout en préservant l'ordre
    unique_paths = []
    seen = set()
    for path in search_paths:
        if path not in seen:
            unique_paths.append(path)
            seen.add(path)
    
    _LOGGER.info(f"Chemins de recherche SD générés pour {original_path}: {unique_paths}")
    return unique_paths


def try_resolve_local_candidate(orig_path: str, sd_path: str) -> Path | None:
    """
    Attempt to find a local copy inside the project dir for build-time processing.
    """
    candidates = []
    try:
        candidates.append(Path(CORE.relative_config_path(sd_path.lstrip("/"))))
    except Exception:
        pass
    try:
        candidates.append(Path(CORE.relative_config_path(orig_path.lstrip("/"))))
    except Exception:
        pass
    try:
        candidates.append(Path(CORE.relative_config_path("sd_card/" + Path(sd_path).name)))
    except Exception:
        pass
    try:
        candidates.append(Path(CORE.relative_config_path("sdcard/" + Path(sd_path).name)))
    except Exception:
        pass

    for c in candidates:
        if c and c.is_file():
            return c
    return None


def generate_sd_image_cpp_class():
    """
    Génère le code C++ pour la classe SDImage qui ne stocke AUCUNE donnée en flash.
    Cette classe charge tout dynamiquement depuis la carte SD au runtime.
    """
    return """
// Classe spécialisée pour images stockées sur carte SD
// AUCUNE donnée stockée en flash memory - tout est chargé dynamiquement
class SDImage : public Image {
 public:
  SDImage(uint32_t width, uint32_t height, ImageType type, TransparencyType transparency,
          const char* const* search_paths, const char* original_path, 
          size_t buffer_size, bool invert_alpha)
      : Image(nullptr, width, height, 1, type),  // data_ = nullptr !
        transparency_(transparency),
        search_paths_(search_paths),
        original_path_(original_path),
        buffer_size_(buffer_size),
        invert_alpha_(invert_alpha),
        data_loaded_(false),
        dynamic_buffer_(nullptr) {}

  virtual ~SDImage() {
    if (dynamic_buffer_) {
      free(dynamic_buffer_);
      dynamic_buffer_ = nullptr;
    }
  }

  // Override get_data_start() - charge depuis SD si nécessaire
  const uint8_t* get_data_start() override {
    if (!data_loaded_) {
      load_from_sd_card();
    }
    return dynamic_buffer_;
  }

  // Override has_transparency() 
  bool has_transparency() override {
    return transparency_ != TRANSPARENCY_OPAQUE;
  }

  // Force un rechargement depuis la SD
  void reload() {
    data_loaded_ = false;
    if (dynamic_buffer_) {
      free(dynamic_buffer_);
      dynamic_buffer_ = nullptr;
    }
  }

 protected:
  bool load_from_sd_card() {
    if (data_loaded_ && dynamic_buffer_) {
      return true;  // Déjà chargé
    }

    // Alloue le buffer dynamiquement - JAMAIS en flash !
    if (!dynamic_buffer_) {
      dynamic_buffer_ = (uint8_t*)malloc(buffer_size_);
      if (!dynamic_buffer_) {
        ESP_LOGE("SDImage", "Impossible d'allouer %zu bytes pour %s", 
                 buffer_size_, original_path_);
        return false;
      }
    }

    // Essaie chaque chemin de recherche jusqu'à trouver l'image
    for (int i = 0; search_paths_[i] != nullptr; i++) {
      const char* path = search_paths_[i];
      
      if (load_and_decode_from_path(path)) {
        ESP_LOGI("SDImage", "Image chargée avec succès: %s", path);
        data_loaded_ = true;
        return true;
      }
    }

    ESP_LOGE("SDImage", "Impossible de charger l'image depuis aucun chemin pour %s", 
             original_path_);
    return false;
  }

  bool load_and_decode_from_path(const char* path) {
    // Cette fonction sera implémentée dans le composant C++
    // Elle doit :
    // 1. Ouvrir le fichier depuis la SD card
    // 2. Decoder l'image (JPEG, PNG, etc.)  
    // 3. Redimensionner à width_ x height_
    // 4. Encoder dans le format target (RGB565, GRAYSCALE, etc.)
    // 5. Stocker dans dynamic_buffer_
    
    // TODO: Implémentation complète avec décodage d'image
    return false;  // Placeholder
  }

 private:
  TransparencyType transparency_;
  const char* const* search_paths_;
  const char* original_path_;
  size_t buffer_size_;
  bool invert_alpha_;
  bool data_loaded_;
  uint8_t* dynamic_buffer_;  // Buffer alloué dynamiquement - JAMAIS en flash !
};
"""


async def write_image(config, all_frames=False):
    """
    Fonction principale de traitement des images avec support complet pour cartes SD.
    Inclut la détection automatique des points de montage.
    ÉVITE COMPLÈTEMENT L'UTILISATION DE LA FLASH MEMORY pour les images SD.
    """
    path_str = config[CONF_FILE]

    # Détecte si c'est une image de la carte SD
    if is_sd_card_path(path_str):
        _LOGGER.info(f"Traitement d'une image SD: {path_str}")
        sd_path = normalize_to_sd_path(path_str)
        search_paths = generate_sd_search_paths(path_str)
        
        # Gestion du resize - OBLIGATOIRE pour les images SD
        if CONF_RESIZE not in config:
            raise cv.Invalid(
                f"Le paramètre 'resize' est obligatoire pour les images de carte SD. "
                f"Spécifiez 'resize: WIDTHxHEIGHT' pour l'image {path_str}"
            )
        
        width, height = config[CONF_RESIZE]
        type = config[CONF_TYPE]
        transparency = config[CONF_TRANSPARENCY]
        invert_alpha = config[CONF_INVERT_ALPHA]
        
        _LOGGER.info(f"Image SD configurée: {path_str} -> {width}x{height}")
        
        def calculate_buffer_size(w, h, img_type, trans):
            """Calcule la taille du buffer pour une configuration donnée"""
            if img_type == "RGB565":
                bpp = 3 if trans == "alpha_channel" else 2
            elif img_type == "RGB":
                bpp = 4 if trans == "alpha_channel" else 3
            elif img_type == "GRAYSCALE":
                bpp = 2 if trans == "alpha_channel" else 1
            elif img_type == "BINARY":
                return ((w + 7) // 8) * h
            else:
                bpp = 3  # Par défaut RGB
            return w * h * bpp
        
        # Calcul de la taille du buffer
        buffer_size = calculate_buffer_size(width, height, type, transparency)
        
        # Validation de la taille - permet des images plus grandes pour ESP32-S3/P4 avec PSRAM
        max_buffer_size = 8 * 1024 * 1024  # 8MB max - ajustable selon votre ESP32
        
        # Avertissement pour les grosses images mais pas d'erreur bloquante
        if buffer_size > 4 * 1024 * 1024:  # > 4MB
            _LOGGER.warning(
                f"Image SD {path_str}: buffer très grand ({buffer_size / (1024*1024):.1f} MB). "
                f"Assurez-vous que votre ESP32 a assez de PSRAM."
            )
        
        if buffer_size > max_buffer_size:
            raise cv.Invalid(
                f"Image SD {path_str}: buffer trop grand ({buffer_size} bytes). "
                f"Maximum autorisé: {max_buffer_size} bytes. "
                f"Réduisez la taille avec resize: ou changez le format."
            )
        
        # Recherche d'un fichier local pour la validation build-time
        local_file = try_resolve_local_candidate(path_str, sd_path)
        if local_file and local_file.is_file():
            _LOGGER.info(f"Fichier local trouvé pour validation: {local_file}")
            
            # Validation du fichier si trouvé localement
            if is_svg_file(local_file):
                validate_cairosvg_installed()
            else:
                try:
                    with Image.open(local_file) as img:
                        orig_width, orig_height = img.size
                        _LOGGER.info(f"Image locale validée: {orig_width}x{orig_height}")
                        
                        # Avertissement si le ratio change beaucoup
                        target_ratio = width / height
                        source_ratio = orig_width / orig_height
                        if abs(target_ratio - source_ratio) > 0.1:
                            _LOGGER.warning(
                                f"Ratio d'aspect changé significativement: "
                                f"{source_ratio:.2f} -> {target_ratio:.2f}"
                            )
                except Exception as e:
                    _LOGGER.warning(f"Impossible de valider l'image locale: {e}")
        else:
            _LOGGER.info(f"Aucun fichier local trouvé pour {path_str}, validation runtime seulement")
        
        # Génère les métadonnées pour le C++
        search_paths_str = ",".join(f'"{path}"' for path in search_paths)
        
        # Code C++ généré - MÉTADONNÉES SEULEMENT, pas de données image
        # Toutes les données seront chargées dynamiquement depuis la SD
        cg.add_global(cg.RawExpression(f"""
// Chemins de recherche SD pour: {path_str} (AUCUNE donnée en flash)
static const char* {config[CONF_ID]}_sd_search_paths[] = {{{search_paths_str}, nullptr}};
"""))

        # Génère la classe SDImage si pas encore fait
        if not hasattr(cg, '_sd_image_class_generated'):
            cg.add_global(cg.RawExpression(generate_sd_image_cpp_class()))
            cg._sd_image_class_generated = True
        
        # AUCUNE donnée en flash memory - tout sera chargé depuis la SD au runtime
        # Utilise la classe SDImage spécialisée qui ne stocke rien en flash
        
        # Création de l'objet SDImage (classe C++ dédiée aux images SD)
        var = cg.new_Pvariable(
            config[CONF_ID],
            SDImage_,                                        # Classe spécialisée SD
            # Paramètres pour SDImage - AUCUN buffer en mémoire flash
            width,                                           # Largeur cible
            height,                                          # Hauteur cible  
            get_image_type_enum(type),                      # Type d'image
            get_transparency_enum(transparency),             # Transparence
            cg.RawExpression(f"{config[CONF_ID]}_sd_search_paths"),  # Chemins de recherche
            cg.RawExpression(f'"{path_str}"'),              # Chemin original pour debug
            buffer_size,                                     # Taille buffer (pour allocation dynamique)
            invert_alpha                                     # Inversion alpha
        )
        
        _LOGGER.info(f"Image SD configurée avec succès: {config[CONF_ID]} - AUCUNE donnée en flash !")
        return var

    else:
        # Traitement standard pour fichiers locaux (code existant)
        return await write_local_image(config, all_frames)


async def write_local_image(config, all_frames=False):
    """
    Traitement des images locales (non-SD) - code original adapté
    """
    path = config[CONF_FILE]
    
    try:
        resize = config.get(CONF_RESIZE)
        if is_svg_file(path):
            validate_cairosvg_installed()
            import cairosvg
            
            if resize:
                req_width, req_height = resize
                svg_image = cairosvg.svg2png(
                    url=path, output_width=req_width, output_height=req_height
                )
            else:
                svg_image = cairosvg.svg2png(url=path)
            image = Image.open(io.BytesIO(svg_image))
        else:
            image = Image.open(path)
            if resize:
                image = image.resize(resize)
        
        frames = []
        if all_frames and hasattr(image, 'n_frames'):
            try:
                for frame_index in range(image.n_frames):
                    image.seek(frame_index)
                    frame = image.copy()
                    frames.append(frame)
            except Exception as e:
                _LOGGER.warning(f"Erreur lors de l'extraction des frames: {e}")
                frames = [image]
        else:
            frames = [image]
        
        # Traitement de chaque frame
        encoded_frames = []
        for frame in frames:
            width, height = frame.size
            if config[CONF_TYPE] == "GRAYSCALE":
                if frame.mode == "LA":
                    pass  # OK
                elif frame.mode in ("RGBA", "RGB"):
                    frame = frame.convert("LA")
                else:
                    frame = frame.convert("L").convert("LA")
            elif frame.mode != "RGBA":
                frame = frame.convert("RGBA")
                
            # Configuration de l'encoder
            encoder = IMAGE_TYPE[config[CONF_TYPE]](
                width,
                height,
                config[CONF_TRANSPARENCY],
                getattr(Image.Dither, config[CONF_DITHER]),
                config[CONF_INVERT_ALPHA]
            )
            
            if byte_order := config.get(CONF_BYTE_ORDER):
                if hasattr(encoder, 'set_big_endian'):
                    encoder.set_big_endian(byte_order == "BIG_ENDIAN")
            
            encoder.convert(frame, path)
            
            # Encodage des pixels
            for y in range(height):
                for x in range(width):
                    encoder.encode(frame.getpixel((x, y)))
                encoder.end_row()
            
            encoded_frames.append(encoder.data)
        
        # Concaténation des frames si multiple
        if len(encoded_frames) > 1:
            all_data = []
            for frame_data in encoded_frames:
                all_data.extend(frame_data)
            data = all_data
        else:
            data = encoded_frames[0]
        
        # Génération du code C++
        rhs = [HexInt(x) for x in data]
        prog_arr = cg.progmem_array(config[CONF_RAW_DATA_ID], rhs)
        
        var = cg.new_Pvariable(
            config[CONF_ID],
            prog_arr,
            encoder.width,
            encoder.height,
            len(encoded_frames),
            get_image_type_enum(config[CONF_TYPE])
        )
        
        return var
        
    except Exception as e:
        _LOGGER.error(f"Erreur lors du traitement de l'image {path}: {e}")
        raise cv.Invalid(f"Impossible de traiter l'image: {e}")


LOCAL_SCHEMA = cv.All(
    {
        cv.Required(CONF_PATH): cv.file_,
    },
    local_path,
)

# Ajout du schéma SD card
SD_CARD_SCHEMA = cv.All(
    {
        cv.Required(CONF_PATH): cv.string,
    },
    sd_card_path,
)


def mdi_schema(source):
    def validate_mdi(value):
        return download_gh_svg(value, source)

    return cv.All(
        cv.Schema(
            {
                cv.Required(CONF_ICON): cv.string,
            }
        ),
        validate_mdi,
    )


WEB_SCHEMA = cv.All(
    {
        cv.Required(CONF_URL): cv.string,
    },
    download_image,
)


TYPED_FILE_SCHEMA = cv.typed_schema(
    {
        SOURCE_LOCAL: LOCAL_SCHEMA,
        SOURCE_WEB: WEB_SCHEMA,
        SOURCE_SD_CARD: SD_CARD_SCHEMA,  # Ajout du schéma SD card
    }
    | {source: mdi_schema(source) for source in MDI_SOURCES},
    key=CONF_SOURCE,
)


def validate_transparency(choices=TRANSPARENCY_TYPES):
    def validate(value):
        if isinstance(value, bool):
            value = str(value)
        return cv.one_of(*choices, lower=True)(value)

    return validate


def validate_type(image_types):
    def validate(value):
        value = cv.one_of(*image_types, upper=True)(value)
        return IMAGE_TYPE[value].validate(value)

    return validate


def validate_settings(value):
    """
    Validate the settings for a single image configuration.
    """
    conf_type = value[CONF_TYPE]
    type_class = IMAGE_TYPE[conf_type]
    transparency = value[CONF_TRANSPARENCY].lower()
    if transparency not in type_class.allow_config:
        raise cv.Invalid(
            f"Image format '{conf_type}' cannot have transparency: {transparency}"
        )
    invert_alpha = value.get(CONF_INVERT_ALPHA, False)
    if (
        invert_alpha
        and transparency != CONF_ALPHA_CHANNEL
        and CONF_INVERT_ALPHA not in type_class.allow_config
    ):
        raise cv.Invalid("No alpha channel to invert")
    if value.get(CONF_BYTE_ORDER) is not None and not callable(
        getattr(type_class, "set_big_endian", None)
    ):
        raise cv.Invalid(
            f"Image format '{conf_type}' does not support byte order configuration"
        )
    if file := value.get(CONF_FILE):
        file_path = str(file)
        
        # Pour les fichiers SD card, on évite la validation locale
        if is_sd_card_path(file_path):
            _LOGGER.info(f"SD card image configured: {file_path}")
            return value
            
        file = Path(file)
        if is_svg_file(file):
            validate_cairosvg_installed()
        else:
            try:
                Image.open(file)
            except UnidentifiedImageError as exc:
                raise cv.Invalid(
                    f"File can't be opened as image: {file.absolute()}"
                ) from exc
    return value


IMAGE_ID_SCHEMA = {
    cv.Required(CONF_ID): cv.declare_id(Image_),
    cv.Required(CONF_FILE): cv.Any(validate_file_shorthand, TYPED_FILE_SCHEMA),
    cv.GenerateID(CONF_RAW_DATA_ID): cv.declare_id(cg.uint8),
}


OPTIONS_SCHEMA = {
    cv.Optional(CONF_RESIZE): cv.dimensions,
    cv.Optional(CONF_DITHER, default="NONE"): cv.one_of(
        "NONE", "FLOYDSTEINBERG", upper=True
    ),
    cv.Optional(CONF_INVERT_ALPHA, default=False): cv.boolean,
    cv.Optional(CONF_BYTE_ORDER): cv.one_of("BIG_ENDIAN", "LITTLE_ENDIAN", upper=True),
    cv.Optional(CONF_TRANSPARENCY, default=CONF_OPAQUE): validate_transparency(),
    cv.Optional(CONF_TYPE): validate_type(IMAGE_TYPE),
}

OPTIONS = [key.schema for key in OPTIONS_SCHEMA]

# image schema with no defaults, used with `CONF_IMAGES` in the config
IMAGE_SCHEMA_NO_DEFAULTS = {
    **IMAGE_ID_SCHEMA,
    **{cv.Optional(key): OPTIONS_SCHEMA[key] for key in OPTIONS},
}

BASE_SCHEMA = cv.Schema(
    {
        **IMAGE_ID_SCHEMA,
        **OPTIONS_SCHEMA,
    }
).add_extra(validate_settings)

IMAGE_SCHEMA = BASE_SCHEMA.extend(
    {
        cv.Required(CONF_TYPE): validate_type(IMAGE_TYPE),
    }
)


def validate_defaults(value):
    """
    Validate the options for images with defaults
    """
    defaults = value[CONF_DEFAULTS]
    result = []
    for index, image in enumerate(value[CONF_IMAGES]):
        type = image.get(CONF_TYPE, defaults.get(CONF_TYPE))
        if type is None:
            raise cv.Invalid(
                "Type is required either in the image config or in the defaults",
                path=[CONF_IMAGES, index],
            )
        type_class = IMAGE_TYPE[type]
        # A default byte order should be simply ignored if the type does not support it
        available_options = [*OPTIONS]
        if (
            not callable(getattr(type_class, "set_big_endian", None))
            and CONF_BYTE_ORDER not in image
        ):
            available_options.remove(CONF_BYTE_ORDER)
        config = {
            **{key: image.get(key, defaults.get(key)) for key in available_options},
            **{key.schema: image[key.schema] for key in IMAGE_ID_SCHEMA},
        }
        validate_settings(config)
        result.append(config)
    return result


def typed_image_schema(image_type):
    """
    Construct a schema for a specific image type, allowing transparency options
    """
    return cv.Any(
        cv.Schema(
            {
                cv.Optional(t.lower()): cv.ensure_list(
                    BASE_SCHEMA.extend(
                        {
                            cv.Optional(
                                CONF_TRANSPARENCY, default=t
                            ): validate_transparency((t,)),
                            cv.Optional(CONF_TYPE, default=image_type): validate_type(
                                (image_type,)
                            ),
                        }
                    )
                )
                for t in IMAGE_TYPE[image_type].allow_config.intersection(
                    TRANSPARENCY_TYPES
                )
            }
        ),
        # Allow a default configuration with no transparency preselected
        cv.ensure_list(
            BASE_SCHEMA.extend(
                {
                    cv.Optional(
                        CONF_TRANSPARENCY, default=CONF_OPAQUE
                    ): validate_transparency(),
                    cv.Optional(CONF_TYPE, default=image_type): validate_type(
                        (image_type,)
                    ),
                }
            )
        ),
    ) 


def _config_schema(config):
    if isinstance(config, list):
        return cv.Schema([IMAGE_SCHEMA])(config)
    if not isinstance(config, dict):
        raise cv.Invalid(
            "Badly formed image configuration, expected a list or a dictionary"
        )
    if CONF_DEFAULTS in config or CONF_IMAGES in config:
        return validate_defaults(
            cv.Schema(
                {
                    cv.Required(CONF_DEFAULTS): OPTIONS_SCHEMA,
                    cv.Required(CONF_IMAGES): cv.ensure_list(IMAGE_SCHEMA_NO_DEFAULTS),
                }
            )(config)
        )
    if CONF_ID in config or CONF_FILE in config:
        return cv.ensure_list(IMAGE_SCHEMA)([config])
    return cv.Schema(
        {cv.Optional(t.lower()): typed_image_schema(t) for t in IMAGE_TYPE}
    )(config)


CONFIG_SCHEMA = _config_schema


def validate_no_flash_memory_usage(config):
    """
    Valide qu'aucune donnée d'image SD ne sera stockée en flash memory.
    Cette fonction s'assure que toutes les images SD sont configurées 
    pour éviter complètement l'utilisation de la flash.
    """
    for image_config in config:
        if is_sd_card_path(image_config.get(CONF_FILE, "")):
            # Vérification que resize est obligatoire
            if CONF_RESIZE not in image_config:
                raise cv.Invalid(
                    f"Les images SD doivent spécifier 'resize:' pour éviter "
                    f"l'utilisation de la flash memory. Image: {image_config[CONF_FILE]}"
                )
            
            # Vérification que le type est compatible avec le chargement dynamique
            img_type = image_config.get(CONF_TYPE)
            if img_type in ["TRANSPARENT_BINARY", "RGB24", "RGBA"]:
                raise cv.Invalid(
                    f"Type d'image '{img_type}' non supporté pour les images SD. "
                    f"Utilisez RGB565, RGB, GRAYSCALE ou BINARY."
                )
            
            _LOGGER.info(
                f"✓ Image SD validée (pas de flash memory): {image_config[CONF_FILE]} "
                f"-> {image_config[CONF_RESIZE]} en {img_type}"
            )
    
    return config


def detect_sd_mount_points():
    """
    Fonction utilitaire pour détecter les points de montage SD disponibles.
    Utilisée pour le debugging et la documentation.
    """
    import os
    
    possible_mounts = [
        "/sdcard", "/sd", "/mnt/sd", "/mnt/sdcard", 
        "/media/sd", "/mnt", "/"
    ]
    
    available_mounts = []
    for mount in possible_mounts:
        if os.path.exists(mount) and os.path.isdir(mount):
            try:
                # Test d'accès en lecture
                os.listdir(mount)
                available_mounts.append(mount)
            except PermissionError:
                # Point de montage existe mais pas d'accès
                available_mounts.append(f"{mount} (no access)")
            except Exception:
                continue
    
    return available_mounts







