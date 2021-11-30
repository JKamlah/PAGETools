from pagetools.src.Page import Page
from pagetools.src.Image import Image, ProcessedImage
from pagetools.src.utils import filesystem
from pagetools.src.utils.constants import predictable_regions
from pagetools.src.utils.page_processing import string_to_coords

from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import List, Iterator, Set, Tuple, Dict

from lxml import etree
from PIL import Image as IMAGE


class Predictor:
    def __init__(self, xml: Path, images: List[Path], include: List[str], exclude: List[str], engine: str, lang: str,
                 out: Path, background, padding: Tuple[int], auto_deskew: bool, deskew: float, pred_index: int,
                 skip_existing: bool):
        self.page = self.xml_to_page(xml)
        self.image = self.get_image(images)

        self.element_list = self.build_element_list(include, exclude)

        self.out = out

        self.engine = engine
        self.lang = lang

        self.background = background
        self.padding = padding

        self.auto_deskew = auto_deskew
        self.deskew = deskew

        self.skip_existing = True

        self.pred_index = pred_index

    @staticmethod
    def xml_to_page(xml: Path):
        return Page(xml)

    @staticmethod
    def get_image(images: List[Path]) -> Image:
        images = [Image(img) for img in images]
        return images[0] if images else []

    @staticmethod
    def build_element_list(include: List[str], exclude: List[str]) -> Set[str]:
        element_list = predictable_regions.copy()
        if "*" in exclude:
            element_list.clear()
        elif exclude:
            for element_type in exclude:
                element_list.remove(element_type)
        if "*" in include:
            element_list = predictable_regions.copy()
        elif include:
            element_list.extend([elem_type for elem_type in include if elem_type != "*"])
        return element_list

    def get_element_data(self) -> OrderedDict:
        element_data = OrderedDict()
        for element_type in self.element_list:
            element_regions = self.page.tree.getroot().findall(f".//page:{element_type}", namespaces=self.page.ns)

            for region in element_regions:
                region_id = region.attrib.get("id")
                if region_id in element_data.keys():
                    continue
                if element_type == "TextLine":
                    orientation = float(region.getparent().attrib.get("orientation", 0))
                else:
                    orientation = float(region.attrib.get("orientation", 0))

                coords = region.find("./page:Coords", namespaces=self.page.ns).attrib["points"]

                text_line_data = {"id": region_id,
                                  "orientation": orientation,
                                  "coords": string_to_coords(coords),
                                  "element": region,
                                  }

                text_equivs = region.findall("./page:TextEquiv", namespaces=self.page.ns)
                if len(text_equivs) > 0:
                    if not self.skip_existing:
                        continue

                element_data[region_id] = text_line_data

        return element_data

    def boundingbox2coords(self, bbox):
        return ((bbox[0], bbox[1]),
                (bbox[0], bbox[3]),
                (bbox[2], bbox[3]),
                (bbox[2], bbox[1]))

    def coords2str(self, coords):
        return ' '.join([f"{point[0]},{point[1]}" for point in coords])

    def coords2pagecoords(self, coords, offset):
        return [(point[0]-self.padding[0]+offset[0], point[1]-self.padding[2]+offset[1]) for point in coords]

    # TODO: Rewrite as soon as PAGEpy is available
    def predict(self):
        if not self.image: return
        data = self.get_element_data()
        img_orig = ProcessedImage(self.image.get_filename(), background=self.background, orientation=0.0)
        if self.engine == 'tesseract':
            from tesserocr import PyTessBaseAPI, RIL, iterate_level
            with PyTessBaseAPI(lang=self.lang, psm=4) as api:
                for element_id, entry in data.items():
                    img = deepcopy(img_orig)
                    img.orientation = entry["orientation"]
                    img.cutout(shape=entry["coords"], padding=self.padding, background=self.background)
                    if self.deskew:
                        img.deskew(self.deskew)
                    elif self.auto_deskew:
                        img.auto_deskew()

                    api.SetImage(IMAGE.fromarray(img.img))
                    api.Recognize()
                    ri = api.GetIterator()
                    line_index = 0
                    fulltext = ''
                    for line in iterate_level(ri, RIL.TEXTLINE):
                        print(f"{element_id}l{line_index}")
                        if not line.Empty(RIL.TEXTLINE):
                            """
                            textline = etree.XML(f"<TextLine id="{entry['id']}l{line_index}" 
                            custom="readingOrder {{index:{line_index};}}">
                            <Coords points="" />
                            <Baseline points="" />
                            <TextEquiv index="{self.pred_index}" conf="{line.Confidence(RIL.TEXTLINE)}">
                            {line.GetUTF8Text(RIL.TEXTLINE)}< Unicode />
                            </TextEquiv >
                            </TextLine >")
                            """
                            print(line.GetUTF8Text(RIL.TEXTLINE).strip())
                            ele_textline = etree.SubElement(entry['element'], 'TextLine',
                                                            {'id': f"{element_id}l{line_index}",
                                                            'custom': f"readingOrder {{index:{line_index};}}"})
                            offset = (min(x[0] for x in entry["coords"]), min(x[1] for x in entry["coords"]))
                            etree.SubElement(ele_textline, 'Coords', {'points': self.coords2str(self.coords2pagecoords(self.boundingbox2coords(line.BoundingBox(RIL.TEXTLINE)), offset))})
                            etree.SubElement(ele_textline, 'Baseline', {'points': self.coords2str(self.coords2pagecoords(line.Baseline(RIL.TEXTLINE),offset))})
                            ele_textequiv = etree.SubElement(ele_textline, 'TextEquiv',
                                                             {'index': str(self.pred_index),
                                                              'conf': str(line.Confidence(RIL.TEXTLINE))})
                            ele_unicode = etree.SubElement(ele_textequiv, 'Unicode')
                            ele_unicode.text = '???' if line.GetUTF8Text(RIL.TEXTLINE).strip() == '' else line.GetUTF8Text(RIL.TEXTLINE).strip()
                            fulltext += ele_unicode.text+'\n'
                            line_index += 1
                        else:
                            print("No text found!")
                    if fulltext != '':
                        ele_textregion = etree.SubElement(entry['element'], 'TextEquiv')
                        ele_unicode = etree.SubElement(ele_textregion, 'Unicode')
                        ele_unicode.text = fulltext

    def export(self, output: Path):
        self.page.export(output)
