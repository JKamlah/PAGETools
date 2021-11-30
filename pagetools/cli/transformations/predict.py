from pagetools.src.utils import filesystem
from pagetools.src.prediction.Predictor import Predictor
from pagetools.src.utils.constants import predictable_regions
from pagetools.src.utils.filesystem import get_suffix

from pathlib import Path
from typing import List, Tuple
import sys
import shutil

import click


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
@click.option("-o", "--output", type=click.Path(path_type=Path), default=Path.cwd(), help="Path where generated files will get saved.")
@click.option("-bg", "--background-color", nargs=3, default=(255, 255, 255), type=int,
              help="RGB color code used to fill up background. Used when padding and / or deskewing.")
@click.option("--background-mode", type=click.Choice(["median", "mean", "dominant"]),
              help="Color calc mode to fill up background (overwrites -bg / --background-color).")
@click.option("-p", "--padding", nargs=4, default=(0, 0, 0, 0), type=int, help="Padding in pixels around the line image"
                                                                        " cutout (top, bottom, left, right). "
                                                                        "Recommend for tesseract : (30, 30, 30, 30)")
@click.option("-ad", "--auto-deskew", is_flag=True, help="Automatically deskew extracted line images using a custom "
                                                         "algorithm (Experimental!).")
@click.option("-d", "--deskew", default=0.0, type=float, help="Angle for manual clockwise rotation of the line images.")
@click.option("-pred", "--pred-index", type=int, default=1, help="Index of the TextEquiv elements containing predicted "
                                                                 "text.")
@click.option("-s/-us", "--safe/--unsafe", default=True, help="Creates backups of original files before overwriting.")
def predict_cli(xmls: List[str], include: List[str], exclude: List[str], skip_existing: bool, image_extension: str,
                engine: str, lang: str, output: Path, background_color: Tuple[int], background_mode: str,
                padding: Tuple[int], pred_index: int, auto_deskew: bool, deskew: float, safe: bool):

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

    click.echo(f"Found {len(file_dict)} PAGE XML files…")

    with click.progressbar(iterable=file_dict.items(), fill_char=click.style("█", dim=True),
                           label="Predicting text lines…") as files:
        for page_idx, (xml, images) in enumerate(files):
            predictor = Predictor(xml, images, include, exclude, engine, lang, output, bg, padding, auto_deskew,
                                  deskew, pred_index, skip_existing)
            predictor.predict()
            if safe:
                shutil.move(xml, Path(xml.parent, xml.stem).with_suffix(f".old{get_suffix(xml)}"))
            predictor.export(xml)

    click.echo(click.style("Text line prediction finished successfully.", fg='green'))



if __name__ == "__main__":
    predict_cli()

