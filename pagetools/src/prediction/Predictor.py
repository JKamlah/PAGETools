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
import cv2
import numpy as np

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
        return [(int(point[0])-self.padding[0]+offset[0], int(point[1])-self.padding[2]+offset[1]) for point in coords]

    @staticmethod
    def fit_baseline2bbox(baseline: List, bbox: List) -> List:
        fitted_baseline = []
        min_y, max_y = (bbox[1], bbox[3]) if bbox[1] < bbox[3] else (bbox[3], bbox[1])
        for baselinepoint in baseline:
            if not (min_y <= baselinepoint[1] <= max_y):
                baselinepoint = (baselinepoint[0], int(abs(max_y - min_y) * 0.33) + min_y)
            if baselinepoint[0] < bbox[0] and not fitted_baseline:
                fitted_baseline.append((bbox[0], baselinepoint[1]))
            elif baselinepoint[0] > bbox[2]:
                if not fitted_baseline:
                    fitted_baseline.extend([(bbox[0], baselinepoint[1]), (bbox[2], baselinepoint[1])])
                else:
                    fitted_baseline.append((bbox[2], baselinepoint[1]))
                break
            else:
                fitted_baseline.append(baselinepoint)
        return fitted_baseline

    @staticmethod
    def fit_baseline2polygon(baseline: List, basic_baseline: List, polygon: List) -> List:
        # Create an upper and lower bound for the polygon and interpolate it to the points of the baseline
        lower_bound = polygon[np.argmax(polygon, axis=0)[0][0] + 1:][::-1]
        lower_bound_xy = list(zip(*[point[0] for point in lower_bound.tolist()]))
        upper_bound = polygon[:np.argmax(polygon, axis=0)[0][0] + 1]
        upper_bound_xy = list(zip(*[point[0] for point in upper_bound.tolist()]))
        min_x, max_x = lower_bound_xy[0][0], lower_bound_xy[0][-1]
        # Convert baseline to points and deduplicate points if needed
        fitted_baseline = []
        sorted_baselinepoints = sorted([list(baselinepoint) for baselinepoints in baseline for baselinepoint in baselinepoints])
        deduplicated_sorted_baselinepoints = [point for idx, point in enumerate(sorted_baselinepoints) if
                                              (idx == 0 or point[0] != sorted_baselinepoints[idx-1][0])]
        #Create an x and y array of the baseline points
        baseline_xy = list(zip(*deduplicated_sorted_baselinepoints))
        basic_baseline_xy = list(zip(*basic_baseline))

        # Add baseline for whole line as fallback
        new_baseline_x = sorted((point for point in baseline_xy[0]+(max_x, min_x) if  min_x <= point <= max_x))

        lower_interp_y = np.interp(new_baseline_x, lower_bound_xy[0], lower_bound_xy[1]).astype(np.int32)
        upper_interp_y = np.interp(new_baseline_x, upper_bound_xy[0], upper_bound_xy[1]).astype(np.int32)

        basic_baseline_interp_y = np.interp(new_baseline_x, basic_baseline_xy[0], basic_baseline_xy[1]).astype(np.int32).tolist()

        # Generate an artificial baseline by 1/3 between upper and lower bound
        midline_y = [np.mean([lower_interp_y[i], upper_interp_y[i]], dtype=np.int32) for i in range(0, len(new_baseline_x))]
        new_baseline_y = [np.mean([midline_y[i], lower_interp_y[i]], dtype=np.int32) for i in range(0, len(new_baseline_x))]

        def get_y_value(x_val, index, basic_baseline_y, new_baseline_y):
            # Check if the basic_baseline has a valid point or take the artificial generated one
            if cv2.pointPolygonTest(polygon, (x_val, basic_baseline_y[index]), False) >= 0:
                    return basic_baseline_y[index]
            else:
                return new_baseline_y[index]

        # Fit the baseline into polygon if necessary take points of the artificial baseline
        for xval_idx, baseline_xval in enumerate(baseline_xy[0]):
            if baseline_xval >= min_x:
                if baseline_xval > max_x:
                    if fitted_baseline[-1][0] <= max_x:
                        fitted_baseline.append((max_x, get_y_value(max_x, -1, basic_baseline_interp_y, new_baseline_y)))
                        break
                elif not fitted_baseline and xval_idx > 0:
                    fitted_baseline.append((baseline_xy[0][xval_idx], get_y_value(new_baseline_x[0], 0, basic_baseline_interp_y, new_baseline_y)))
                elif cv2.pointPolygonTest(polygon, (baseline_xy[0][xval_idx], baseline_xy[1][xval_idx]), False) < 0:
                    fitted_baseline.append((baseline_xy[0][xval_idx], get_y_value(baseline_xy[0][xval_idx], new_baseline_x.index(baseline_xval), basic_baseline_interp_y, new_baseline_y)))
                else:
                    fitted_baseline.append((baseline_xy[0][xval_idx], baseline_xy[1][xval_idx]))
            elif xval_idx == 0 and baseline_xy[0][xval_idx+1] > new_baseline_x[-1]:
                fitted_baseline.append((new_baseline_y[0], new_baseline_y[0]))
        return fitted_baseline

    def tesseract_symbol_bboxs_and_baselines(self, ri, RIL, iterate_level):
        bbox, bboxs = [], []
        baselines, baseline = [], []
        for symbol_idx, symbol in enumerate(iterate_level(ri, RIL.SYMBOL)):
            if symbol.IsAtBeginningOf(RIL.TEXTLINE) and symbol_idx != 0:
                bboxs.append(bbox)
                baselines.append(baseline)
                bbox, baseline = [], []
            if not symbol.Empty(RIL.SYMBOL):
                bbox.append(symbol.BoundingBoxInternal(RIL.SYMBOL))
                if symbol.Baseline(RIL.SYMBOL) not in baseline:
                    baseline.append(symbol.Baseline(RIL.SYMBOL))
        else:
            bboxs.append(bbox)
            baselines.append(baseline)
        ri.RestartRow()
        ri.Begin()
        return bboxs, baselines

    def shrink_bboxs(self, bboxs: list) -> list:
        if len(bboxs) > 1 and abs(bboxs[0][2] - bboxs[0][0]) > abs(bboxs[0][1] - bboxs[0][3]) * 2:
            bboxs.pop(0)
        if len(bboxs) > 1 and abs(bboxs[-1][2] - bboxs[-1][0]) > abs(bboxs[-1][1] - bboxs[-1][3]) * 2:
            bboxs.pop(-1)
        return bboxs

    def polygon_from_bboxs(self, bboxs):
        # Calculate polygon
        points = [coords for bbox in bboxs for coords in self.boundingbox2coords(bbox)]
        polygon = cv2.convexHull(np.array(points, dtype=np.int32), False)
        return polygon

    def fit_line_to_region_polygon(self, line_polygon, region_polygon):
        fitted_polygon_area, fitted_polygon = cv2.intersectConvexConvex(line_polygon, region_polygon)
        return fitted_polygon_area, fitted_polygon.astype(np.int32)

    def rotate_polygon_to_start_from_left_top(self, polygon):
        min_idx = np.argmin(polygon, axis=0)[0][0]
        if min_idx != 0 or polygon[0][0][0] != polygon[0][0][-1]:
            polygon = np.concatenate((polygon[min_idx + 1:], polygon[:min_idx + 1]), axis=0)
        return polygon

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

                    symbol_bboxs, symbol_baselines = self.tesseract_symbol_bboxs_and_baselines(ri, RIL, iterate_level)

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
                            offset = (min(x[0] for x in entry["coords"]), min(x[1] for x in entry["coords"]))
                            baseline = line.Baseline(RIL.TEXTLINE)
                            linetext = '���' if line.GetUTF8Text(RIL.TEXTLINE).strip() == '' else line.GetUTF8Text(RIL.TEXTLINE).strip()
                            # Skip unrecognized textlines in the block end
                            if linetext == '���' and lidx != 0 and lidx == len(symbol_bboxs)-1:
                                continue
                            # The basic bbox list(line.BoundingBox(RIL.TEXTLINE)) is currently not used
                            # Experimental find bbox which are to wide (use only for horizontal text)
                            bboxs = self.shrink_bboxs(symbol_bboxs[lidx])
                            # Add missing points
                            polygon = self.polygon_from_bboxs(bboxs)
                            poly_area, polygon = self.fit_line_to_region_polygon(polygon, np.array([[self.padding[0], self.padding[1]],
                                                                                         [self.padding[0], img.img.shape[0]-self.padding[1]],
                                                                                         [img.img.shape[1]-self.padding[0], img.img.shape[0]-self.padding[1]],
                                                                                         [img.img.shape[1]-self.padding[0], self.padding[1]]], dtype=np.int32))
                            polygon = self.rotate_polygon_to_start_from_left_top(polygon)
                            # The basic_baseline matching (fit_baseline2bbox(baseline, bbox)) is currently not used
                            # Fitting the baseline into the polygon
                            baseline = self.fit_baseline2polygon(symbol_baselines[lidx], baseline, polygon)
                            ele_textline = etree.SubElement(entry['element'], 'TextLine',
                                                            {'id': f"{element_id}l{line_index}",
                                                            'custom': f"readingOrder {{index:{line_index};}}"})
                            etree.SubElement(ele_textline, 'Coords', {'points': self.coords2str(self.coords2pagecoords([x[0] for x in polygon.tolist()], offset))})
                            etree.SubElement(ele_textline, 'Baseline', {'points': self.coords2str(self.coords2pagecoords(baseline, offset))})
                            # Writing TextEquiv parameters but index and conf aren't added currently
                            # not included: {'index': str(self.pred_index), 'conf': str(line.Confidence(RIL.TEXTLINE))})
                            ele_textequiv = etree.SubElement(ele_textline, 'TextEquiv')
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
