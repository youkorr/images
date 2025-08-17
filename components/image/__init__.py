from __future__ import annotations

import hashlib
import io
import logging
from pathlib import Path
import re

from PIL import Image, UnidentifiedImageError

import esphome.codegen as cg
from esphome import core, external_files
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
DEPENDENCIES = ["display"]

# Configuration pour la liaison avec sd_mmc_card
CONF_SD_MMC_CARD_ID = "sd_mmc_card_id"

image_ns = cg.esphome_ns.namespace("image")
ImageType = image_ns.enum("ImageType")

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
    full_path = "sd://" + value  # Conserve le préfixe 'sd://'
    _LOGGER.info(f"Chemin SD résolu: {full_path}")
    return full_path


def is_sd_card_path(path_str: str) -> bool:
    """Check if a path is an SD card path"""
    if not isinstance(path_str, str):
        return False
    path_str = path_str.strip()
    return path_str.startswith("sd://")


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

    # Vérification pour les chemins SD card
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


async def write_image(config, all_frames=False):
    """Fonction principale de traitement des images."""
    path_str = config[CONF_FILE]

    # Détecte si c'est une image de la carte SD
    if is_sd_card_path(path_str):
        _LOGGER.info(f"Traitement d'une image SD: {path_str}")
        sd_path = sd_card_path(path_str)

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

        # Crée un buffer de la bonne taille rempli de zéros
        # Ceci est un placeholder - l'image réelle sera chargée au runtime
        placeholder_data = [0] * buffer_size

        # Configuration de l'encoder pour les métadonnées
        encoder = IMAGE_TYPE[type](width, height, transparency,
                                   Image.Dither.NONE, invert_alpha)

        if byte_order := config.get(CONF_BYTE_ORDER):
            if hasattr(encoder, "set_big_endian"):
                encoder.set_big_endian(byte_order == "BIG_ENDIAN")
                _LOGGER.info(f"Configuration byte_order pour {path_str}: {byte_order}")

        rhs = [HexInt(x) for x in placeholder_data]
        prog_arr = cg.progmem_array(config[CONF_RAW_DATA_ID], rhs)
        image_type = get_image_type_enum(type)
        trans_value = get_transparency_enum(transparency)

        _LOGGER.info(f"""Configuration de l'image SD:
            - Fichier: {sd_path}
            - Dimensions: {width}x{height}
            - Type: {type}
            - Transparence: {transparency}
            - Buffer size: {buffer_size} bytes ({buffer_size / 1024:.1f} KB)
        """)

        return prog_arr, width, height, image_type, trans_value, 1, True, sd_path

    # Code existant pour les images normales (non-SD)
    path = Path(path_str)
    if not path.is_file():
        raise core.EsphomeError(f"Impossible de charger le fichier image {path}")

    resize = config.get(CONF_RESIZE)
    if is_svg_file(path):
        from cairosvg import svg2png
        if not resize:
            resize = (None, None)
        with open(path, "rb") as file:
            image = svg2png(
                file_obj=file,
                output_width=resize[0],
                output_height=resize[1],
            )
        image = Image.open(io.BytesIO(image))
        width, height = image.size
    else:
        image = Image.open(path)
        width, height = image.size
        if resize:
            ratio = min(
                min(width, resize[0]) / width,
                min(height, resize[1]) / height
            )
            width, height = int(width * ratio), int(height * ratio)

    if not resize and (width > 500 or height > 500):
        _LOGGER.warning(
            'L\'image "%s" est très grande. Considérez utiliser le paramètre resize.',
            path,
        )

    dither = (
        Image.Dither.NONE
        if config[CONF_DITHER] == "NONE"
        else Image.Dither.FLOYDSTEINBERG
    )
    type = config[CONF_TYPE]
    transparency = config[CONF_TRANSPARENCY]
    invert_alpha = config[CONF_INVERT_ALPHA]
    frame_count = 1

    if all_frames:
        try:
            frame_count = image.n_frames
        except AttributeError:
            pass
        if frame_count <= 1:
            _LOGGER.warning("Le fichier image %s n'a pas de frames d'animation", path)

    total_rows = height * frame_count
    encoder = IMAGE_TYPE[type](width, total_rows, transparency, dither, invert_alpha)

    if byte_order := config.get(CONF_BYTE_ORDER):
        if hasattr(encoder, "set_big_endian"):
            encoder.set_big_endian(byte_order == "BIG_ENDIAN")

    for frame_index in range(frame_count):
        image.seek(frame_index)
        pixels = encoder.convert(image.resize((width, height)), path).getdata()
        for row in range(height):
            for col in range(width):
                encoder.encode(pixels[row * width + col])
            encoder.end_row()

    rhs = [HexInt(x) for x in encoder.data]
    prog_arr = cg.progmem_array(config[CONF_RAW_DATA_ID], rhs)
    image_type = get_image_type_enum(type)
    trans_value = get_transparency_enum(transparency)

    return prog_arr, width, height, image_type, trans_value, frame_count, False, None


async def to_code(config):
    if isinstance(config, list):
        for entry in config:
            await to_code(entry)
    elif CONF_ID not in config:
        for entry in config.values():
            await to_code(entry)
    else:
        prog_arr, width, height, image_type, trans_value, _, sd_runtime, sd_path = await write_image(config)

        var = cg.new_Pvariable(config[CONF_ID], prog_arr, width, height, image_type, trans_value)

        # Si image configurée pour lecture runtime depuis la SD
        if sd_runtime:
            cg.add(var.set_sd_path(sd_path))
            cg.add(var.set_sd_runtime(True))
            _LOGGER.info(f"Image {config[CONF_ID]} configured for SD card runtime loading: {sd_path}")

            # Ajouter seulement le define nécessaire
            cg.add_define("USE_SD_CARD_IMAGES")







