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
                (bbox[2], bbox[1]),
                (bbox[2], bbox[3]),
                (bbox[0], bbox[3]))

    def coords2str(self, coords):
        return ' '.join([f"{point[0]},{point[1]}" for point in coords])

    def coords2pagecoords(self, coords, offset):
        return [(point[0]-self.padding[0]+offset[0], point[1]-self.padding[2]+offset[1]) for point in coords]

    @staticmethod
    def fit_baseline2bbox(baseline: List, bbox: List) -> List:
        fitted_baseline = []
        for baselinepoint in baseline:
            if not (bbox[1] <= baselinepoint[1] <= bbox[3]):
                baselinepoint = (baselinepoint[0], int((bbox[3]-bbox[1])*0.67)+bbox[1])
            if baselinepoint[0] < bbox[0] and not fitted_baseline:
                fitted_baseline.append((bbox[0], baselinepoint[1]))
            elif baselinepoint[0] > bbox[2]:
                if not fitted_baseline:
                    fitted_baseline.extend([(bbox[0], baselinepoint[1]),(bbox[2], baselinepoint[1])])
                else:
                    fitted_baseline.append((bbox[2], baselinepoint[1]))
                break
            else:
                fitted_baseline.append(baselinepoint)
        return fitted_baseline

    def shrink_bbox(self, bbox: list, symbol_bbox: list) -> list:
        if len(symbol_bbox) > 1:
            if abs(symbol_bbox[0][2] - symbol_bbox[0][0]) > abs(symbol_bbox[0][1] - symbol_bbox[0][3]) * 2:
                bbox[0] = symbol_bbox[1][0]
            elif bbox[0] < (symbol_bbox[0][0]-abs(symbol_bbox[-1][1] - symbol_bbox[-1][3])*2):
                bbox[0] = symbol_bbox[0][0]
            if abs(symbol_bbox[-1][2] - symbol_bbox[-1][0]) > abs(symbol_bbox[-1][1] - symbol_bbox[-1][3]) * 2:
                bbox[2] = symbol_bbox[-2][2]
            elif bbox[2] > (symbol_bbox[-1][2]+abs(symbol_bbox[-1][1] - symbol_bbox[-1][3])*2):
                bbox[2] = symbol_bbox[-1][2]
        return bbox

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
                    symbol_bbox, symbol_bboxs = [], []
                    for symbol_idx, symbol in enumerate(iterate_level(ri, RIL.SYMBOL)):
                        if symbol.IsAtBeginningOf(RIL.TEXTLINE) and symbol_idx != 0:
                            if len(symbol_bbox) > 4:
                                symbol_bbox = symbol_bbox[0:2]+symbol_bbox[-2:]
                            symbol_bboxs.append(symbol_bbox)
                            symbol_bbox = []
                        if not symbol.Empty(RIL.SYMBOL):
                            symbol_bbox.append(symbol.BoundingBoxInternal(RIL.SYMBOL))
                    else:
                        if len(symbol_bbox) > 4:
                            symbol_bbox = symbol_bbox[0:2] + symbol_bbox[-2:]
                        symbol_bboxs.append(symbol_bbox)
                    ri.RestartRow()
                    ri.Begin()
                    for lidx, line in enumerate(iterate_level(ri, RIL.TEXTLINE)):
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
                            if line.GetUTF8Text(RIL.TEXTLINE).strip() == 'die Zinsſcheinne':
                                stop = 1
                            ele_textline = etree.SubElement(entry['element'], 'TextLine',
                                                            {'id': f"{element_id}l{line_index}",
                                                            'custom': f"readingOrder {{index:{line_index};}}"})
                            offset = (min(x[0] for x in entry["coords"]), min(x[1] for x in entry["coords"]))
                            bbox = list(line.BoundingBox(RIL.TEXTLINE))
                            baseline = line.Baseline(RIL.TEXTLINE)
                            linetext = '���' if line.GetUTF8Text(RIL.TEXTLINE).strip() == '' else line.GetUTF8Text(RIL.TEXTLINE).strip()

                            # Experimental find bbox which are to wide (use only for horizontal text)
                            bbox = self.shrink_bbox(bbox, symbol_bboxs[lidx])
                            baseline = self.fit_baseline2bbox(baseline, bbox)

                            etree.SubElement(ele_textline, 'Coords', {'points': self.coords2str(self.coords2pagecoords(self.boundingbox2coords(bbox), offset))})
                            etree.SubElement(ele_textline, 'Baseline', {'points': self.coords2str(self.coords2pagecoords(baseline, offset))})
                            ele_textequiv = etree.SubElement(ele_textline, 'TextEquiv')#,
                                                             #{'index': str(self.pred_index),
                                                              #'conf': str(line.Confidence(RIL.TEXTLINE))})
                            ele_unicode = etree.SubElement(ele_textequiv, 'Unicode')
                            ele_unicode.text = linetext
                            fulltext += ele_unicode.text+'\n'
                            line_index += 1
                        else:
                            print("No text found!")
                    # Does only TextRegion need a fulltext summary of TextLines?
                    if fulltext != '' and entry['element'].tag.rsplit('}', 1)[-1] in ['TextRegion']:
                        ele_textregion = etree.SubElement(entry['element'], 'TextEquiv')
                        ele_unicode = etree.SubElement(ele_textregion, 'Unicode')
                        ele_unicode.text = fulltext

    def export(self, output: Path):
        self.page.export(output)
