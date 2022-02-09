import shutil
import sys
from itertools import repeat
from multiprocessing import Pool
from pathlib import Path
from typing import List, Tuple

import click

from pagetools.src.prediction.Predictor import Predictor
from pagetools.src.utils import filesystem
from pagetools.src.utils.constants import predictable_regions
from pagetools.src.utils.filesystem import get_suffix

available_regions = predictable_regions.copy()
available_regions.append("*")


@click.command("predict", help="Predicts TextEquivs for region elements in files.")
@click.argument("xmls", nargs=-1, required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--include", multiple=True, type=click.Choice(available_regions, case_sensitive=False),
              help="PAGE XML element types to extract (highest priority).")
@click.option("--exclude", multiple=True, type=click.Choice(available_regions, case_sensitive=False),
              help="PAGE XML element types to exclude from extraction (lowest priority).")
@click.option("-se", "--skip-existing", is_flag=True, type=bool, default=False,
              help="Skips existing Textequiv with pred-index")
@click.option("-ie", "--image-extension", default=".png", type=str, help="Extension of image files. Must be in the same"
                                                                         " directory as corresponding XML file.")
# Calamari to come?
@click.option("-e", "--engine", default="tesseract", type=click.Choice(["tesseract"], case_sensitive=False),
              help="OCR-Engine to find lines and predict text.")
@click.option("-l", "--lang", type=str, default="eng", help="Modelname/Checkpoint to predict.")
@click.option("-psm", default=[4, 7], type=int, multiple=True,
              help="Pagesegementation modes to process the area. If one fails the next psm will be used (4, 7)")
@click.option("-placeholder", "--textline-placeholder", is_flag=True,
              help="If no text is found in the region an textline placholder will be created")
@click.option("-o", "--output", type=click.Path(path_type=Path), default=Path.cwd(),
              help="Path where generated files will get saved.")
@click.option("-bg", "--background-color", nargs=3, default=(255, 255, 255), type=int,
              help="RGB color code used to fill up background. Used when padding and / or deskewing.")
@click.option("--background-mode", type=click.Choice(["median", "mean", "dominant"]),
              help="Color calc mode to fill up background (overwrites -bg / --background-color).")
@click.option("-p", "--padding", nargs=4, default=(0, 0, 0, 0), type=int, help="Padding in pixels around the line image"
                                                                               " cutout (top, bottom, left, right). "
                                                                               " Recommend for tesseract : "
                                                                               "(30, 30, 30, 30)")
@click.option("-ad", "--auto-deskew", is_flag=True, help="Automatically deskew extracted line images using a custom "
                                                         "algorithm (Experimental!).")
@click.option("-d", "--deskew", default=0.0, type=float, help="Angle for manual clockwise rotation of the line images.")
@click.option("-w", "--worker", default=None, type=int, help="Worker for multiprocessing.")
@click.option("-pred", "--pred-index", default=1, type=int, help="Index of the TextEquiv elements containing predicted "
                                                                 "text.")
@click.option("-s/-us", "--safe/--unsafe", default=True, help="Creates backups of original files before overwriting.")
def predict_cli(xmls: List[Path], include: List[str], exclude: List[str], skip_existing: bool, image_extension: str,
                engine: str, lang: str, psm: Tuple[int], textline_placeholder: bool, output: Path,
                background_color: Tuple[int], background_mode: str, padding: Tuple[int], worker: int, pred_index: int,
                auto_deskew: bool, deskew: float, safe: bool):

    if engine == 'tesseract' and 'tesserocr' in sys.modules:
        "Please install tesserocr (with wheels) to use tesseract engine"
        return

    file_dict = filesystem.collect_files(map(Path, xmls), image_extension)

    if not file_dict:
        click.echo(click.style("No XML files found.\nAborting…", fg='red'))
        return

    if background_mode:
        bg = ("calculate", background_mode)
    else:
        bg = ("color", background_color)

    with Pool(processes=worker) as pool:
        pool.starmap(predict, zip(file_dict.keys(), file_dict.values(), repeat(include), repeat(exclude),
                                  repeat(skip_existing), repeat(engine), repeat(lang), repeat(psm),
                                  repeat(textline_placeholder), repeat(output), repeat(bg), repeat(padding),
                                  repeat(pred_index), repeat(auto_deskew), repeat(deskew), repeat(safe)))


def predict(xml: Path, images: List[Path], include: List[str], exclude: List[str], skip_existing: bool,
            engine: str, lang: str, psm: Tuple[int], textline_placeholder: bool, output: Path,
            bg: Tuple[str], padding: Tuple[int], pred_index: int, auto_deskew: bool, deskew: float, safe: bool):
    """ Multiprocessable prediction function """
    if not any(images) or \
            '.old' in xml.name or Path(xml.parent, xml.stem).with_suffix(f".old{get_suffix(xml)}").exists():
        print(f"{xml.with_suffix('').name}: prediction cancelled --> no image available")
        return
    predictor = Predictor(xml, images, include, exclude, engine, lang, psm, textline_placeholder, output, bg,
                          padding, auto_deskew, deskew, pred_index, skip_existing)
    print(f"{xml.with_suffix('').name}: start prediction with {engine} ")
    predictor.predict()
    if safe:
        shutil.move(xml, Path(xml.parent, xml.stem).with_suffix(f".old{get_suffix(xml)}"))
    predictor.export(xml)


if __name__ == "__main__":
    predict_cli()
