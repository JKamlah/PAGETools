from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL.Image import fromarray
from lxml import etree
from shapely.affinity import scale, translate, rotate, interpret_origin
from shapely.geometry import Polygon, Point, LineString, MultiPoint
from shapely.ops import split, nearest_points

from pagetools.src.Image import Image, ProcessedImage
from pagetools.src.Page import Page
from pagetools.src.utils.constants import predictable_regions
from pagetools.src.utils.page_processing import string_to_coords


class Predictor:
    def __init__(self, xml: Path, images: List[Path], include: List[str], exclude: List[str], engine: str, lang: str,
                 psm: Tuple[int], textline_placeholder: bool, text_wildcard: str, delete_empty_textlines: bool,
                 out: Path, background, padding: Tuple[int], auto_deskew: bool, deskew: float,
                 pred_index: int, source_index: int, skip_existing: bool):
        self.xml = xml
        self.page = self.xml_to_page(xml)
        self.image = self.get_image(images)

        self.element_list = self.build_element_list(include, exclude)

        self.out = out
        self.engine = engine
        self.lang = lang
        self.psm = psm

        self.textline_placeholder = textline_placeholder
        self.text_wildcard = text_wildcard
        self.delete_empty_textlines = delete_empty_textlines

        self.background = background
        self.padding = padding

        self.auto_deskew = auto_deskew
        self.deskew = deskew

        self.skip_existing = skip_existing

        self.pred_index = pred_index
        self.source_index = source_index
        self.text_equiv_attrib = {'index': str(self.pred_index)}

    @staticmethod
    def xml_to_page(xml: Path):
        """ Create a page object """
        return Page(xml)

    @staticmethod
    def get_image(images: List[Path]) -> Image:
        """ Load image as pil object by filename """
        images = [Image(img) for img in images]
        return images[0] if images else []

    @staticmethod
    def build_element_list(include: List[str], exclude: List[str]) -> Tuple[str]:
        """ Create a list of all elementtypes to be processed """
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
        return tuple(element_list)

    def get_element_data(self) -> OrderedDict:
        """ Collect all element to be processed """
        element_data = OrderedDict()
        self.change_index()
        for element_type in self.element_list:
            element_regions = self.page.tree.getroot().findall(f".//page:{element_type}", namespaces=self.page.ns)
            for region in element_regions:
                # Skip parent elements with child elements which are also in the element list
                ele_with_id = region.findall('.//*[@id]', namespaces=self.page.ns)
                if ele_with_id and ele_with_id[0].tag.endswith(self.element_list):
                    continue
                region_id = region.attrib.get("id")
                if region_id in element_data.keys():
                    continue
                if element_type == "TextLine":
                    if region.find(f"./page:TextEquiv[@index='{self.source_index}']", namespaces=self.page.ns) is None:
                        continue
                    orientation = float(region.getparent().attrib.get("orientation", 0))
                else:
                    orientation = float(region.attrib.get("orientation", 0))

                coords = region.find("./page:Coords", namespaces=self.page.ns).attrib["points"]

                text_line_data = {"id": region_id,
                                  "type": element_type,
                                  "orientation": orientation,
                                  "coords": string_to_coords(coords),
                                  "element": region}

                if element_type != "TextLine" and region.findall("./page:TextEquiv",
                                                                  namespaces=self.page.ns) is not None:
                    if self.skip_existing and region.find(f"./page:TextEquiv[@index='{self.pred_index}']",
                                                          namespaces=self.page.ns) is not None:
                        continue

                element_data[region_id] = text_line_data

        return element_data

    def change_index(self):
        # Delete existing text equivs with the same pred_index and give text equivs with no index The pred_index +1
        for text_equiv in self.page.get_text_equivs():
            if text_equiv.get("index") == self.pred_index and not self.skip_existing:
                text_equiv.getparent().remove(text_equiv)
            elif text_equiv.get("index") is None:
                self.source_index = self.pred_index if self.skip_existing else self.pred_index + 1
                text_equiv.set("index", str(self.source_index))

    @staticmethod
    def boundingbox2coords(bbox):
        """ Create coordinates of bounding boxes """
        return ((bbox[0], bbox[1]),
                (bbox[2], bbox[1]),
                (bbox[2], bbox[3]),
                (bbox[0], bbox[3]))

    @staticmethod
    def coords2str(coords):
        """ Create a string representation of the coordinates """
        return ' '.join([f"{int(np.around(point[0], decimals=1))},{int(np.around(point[1], decimals=1))}"
                         for point in coords])

    @staticmethod
    def fit_baseline2polygon(baseline: List, basic_baseline: List, line_polygon: Polygon) -> LineString:
        """ Fit a baseline into the line-polygon based on the existing baseline and an artificial baseline """
        # Create an upper and lower bound for the polygon and interpolate it to the points of the baseline
        boundary = np.asarray(line_polygon.exterior)
        # Deduplicate
        _, idx = np.unique(boundary, return_index=True, axis=0)
        boundary = boundary[np.sort(idx)]
        # Get splitting point
        splitting_point_idx = np.argmax(boundary, axis=0)[0]
        # Check if splitting point is above the centroid else take the next polygon as splitting point
        if not 2 < splitting_point_idx < len(boundary) - 2 and \
                boundary[splitting_point_idx][1] > list(line_polygon.exterior.centroid.coords)[0][1]:
            splitting_point_idx -= 1
        # Create a lower and upper boundary based on the splitting point
        lower_boundary = boundary[splitting_point_idx + 1:][::-1]
        lower_boundary_xy = list(zip(*[point for point in lower_boundary.tolist()]))
        upper_boundary = boundary[:splitting_point_idx + 1]
        upper_boundary_xy = list(zip(*[point for point in upper_boundary.tolist()]))
        min_x, max_x = lower_boundary_xy[0][0], lower_boundary_xy[0][-1]

        # Convert baseline to points and deduplicate points if needed
        fitted_baseline = []
        sorted_baselinepoints = sorted([list(baselinepoint) for baselinepoints in baseline
                                        for baselinepoint in baselinepoints])
        deduplicated_sorted_baselinepoints = [point for idx, point in enumerate(sorted_baselinepoints) if
                                              (idx == 0 or point[0] != sorted_baselinepoints[idx - 1][0])]
        baseline_poly = LineString(deduplicated_sorted_baselinepoints)

        if line_polygon.covers(baseline_poly):
            return baseline_poly

        if len(lower_boundary_xy[0]) < 2 and len(upper_boundary) < 2:
            centroid_pt = list(line_polygon.centroid.coords)[0]
            return LineString([centroid_pt, centroid_pt])

        # Create an x and y array of the baseline points
        baseline_xy = list(zip(*deduplicated_sorted_baselinepoints))
        if basic_baseline:
            basic_baseline_xy = list(zip(*basic_baseline))
        else:
            basic_baseline_xy = baseline_xy

        # Add baseline for whole line as fallback
        new_baseline_x = sorted((point for point in set(baseline_xy[0] + (max_x, min_x)) if min_x <= point <= max_x))

        lower_interp_y = np.interp(new_baseline_x, lower_boundary_xy[0], lower_boundary_xy[1]).astype(np.int32)
        upper_interp_y = np.interp(new_baseline_x, upper_boundary_xy[0], upper_boundary_xy[1]).astype(np.int32)

        basic_baseline_interp_y = np.interp(new_baseline_x, basic_baseline_xy[0],
                                            basic_baseline_xy[1]).astype(np.int32).tolist()

        # Generate an artificial baseline by 1/3 between upper and lower boundary
        midline_y = [np.mean([lower_interp_y[i], upper_interp_y[i]], dtype=np.int32)
                     for i in range(0, len(new_baseline_x))]
        new_baseline_y = [np.mean([midline_y[i], lower_interp_y[i]], dtype=np.int32)
                          for i in range(0, len(new_baseline_x))]

        def get_y_value(x_val, index, basic_baseline_y, new_baseline_y):
            """ Check if the basic_baseline has a valid point or take the artificial generated one """
            if line_polygon.covers(Point(x_val, basic_baseline_y[index])):
                return basic_baseline_y[index]
            else:
                return new_baseline_y[index]

        # Fit the baseline into polygon if necessary take points of the artificial baseline
        for coords_idx, (x_val, y_val) in enumerate(baseline_poly.coords):
            if x_val >= min_x:
                if x_val > max_x:
                    if not len(fitted_baseline) > 0:
                        fitted_baseline.append((min_x, get_y_value(min_x, -1, basic_baseline_interp_y, new_baseline_y)))
                    if fitted_baseline[-1][0] <= max_x:
                        fitted_baseline.append((max_x, get_y_value(max_x, -1, basic_baseline_interp_y, new_baseline_y)))
                        break
                elif not fitted_baseline and coords_idx > 0:
                    fitted_baseline.append((baseline_xy[0][coords_idx], get_y_value(new_baseline_x[0], 0,
                                                                                    basic_baseline_interp_y,
                                                                                    new_baseline_y)))
                elif not line_polygon.covers(Point(baseline_xy[0][coords_idx], baseline_xy[1][coords_idx])):
                    fitted_baseline.append((baseline_xy[0][coords_idx], get_y_value(baseline_xy[0][coords_idx],
                                                                                    new_baseline_x.index(x_val),
                                                                                    basic_baseline_interp_y,
                                                                                    new_baseline_y)))
                else:
                    fitted_baseline.append((baseline_xy[0][coords_idx], baseline_xy[1][coords_idx]))
            elif coords_idx == 0 and baseline_xy[0][coords_idx + 1] > new_baseline_x[-1]:
                fitted_baseline.append((new_baseline_x[0], new_baseline_y[0]))
        fitted_baseline = [point for idx, point in enumerate(fitted_baseline) if
                           (idx == 0 or point[0] != fitted_baseline[idx - 1][0])]
        if len(fitted_baseline) < 2:
            fitted_baseline = [(min_x, get_y_value(min_x, -1, basic_baseline_interp_y, new_baseline_y)),
                               (max_x, get_y_value(max_x, -1, basic_baseline_interp_y, new_baseline_y))]
        for pt_idx in [0, -1]:
            if not line_polygon.covers(Point(*fitted_baseline[pt_idx])):
                fitted_baseline[pt_idx] = list(nearest_points(line_polygon,
                                                              Point(*fitted_baseline[pt_idx]))[1].coords)[0]

        fitted_baseline = LineString(fitted_baseline)

        if not line_polygon.covers(fitted_baseline):
            try:
                for fitted_baseline_split in split(fitted_baseline, line_polygon):
                    if line_polygon.covers(fitted_baseline_split):
                        fitted_baseline = fitted_baseline_split
                        break
                else:
                    fitted_baseline = line_polygon.intersection(fitted_baseline)
            except:
                # If there is no other possibility take the first two points of the lower_boundary or line_polygon
                if len(lower_boundary) > 1:
                    fitted_baseline = LineString(lower_boundary[:2])
                else:
                    fitted_baseline = LineString(line_polygon.boundary.coords[:2])
        return fitted_baseline

    @staticmethod
    def tesseract_symbol_bboxs_and_baselines(ri, RIL, iterate_level):
        """ Collect all bboxs and baseline values on symbol/glyph level and reset the iterator"""
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

    @staticmethod
    def shrink_bboxs(bboxs: list) -> list:
        """ Correcting tesseracts tendency to overextend the bbox due to dirt on the page  """
        if len(bboxs) > 1 and abs(bboxs[0][2] - bboxs[0][0]) > abs(bboxs[0][1] - bboxs[0][3]) * 2:
            bboxs.pop(0)
        if len(bboxs) > 1 and abs(bboxs[-1][2] - bboxs[-1][0]) > abs(bboxs[-1][1] - bboxs[-1][3]) * 2:
            bboxs.pop(-1)
        return bboxs

    def polygon_from_bboxs(self, bboxs):
        """ Create a multipoint object based on single bbox points and calculate the convex hull """
        return MultiPoint([coords for bbox in bboxs for coords in self.boundingbox2coords(bbox)]).convex_hull

    @staticmethod
    def fit_line_to_region_polygon(line_polygon, region_polygon):
        """ Calculate the convex hull of the intersection of line-polygon and region-polygon """
        intersection = line_polygon.intersection(region_polygon)
        return intersection.convex_hull

    @staticmethod
    def rotate_polygon_to_start_from_left_top(line_polygon):
        """ Set the upper left point as starting point and make the rotation clockwise """
        ring = np.asarray(list(line_polygon.exterior.coords)[::-1])
        min_idx = np.argmin(ring, axis=0)[0]
        if ring[min_idx][1] < list(line_polygon.exterior.centroid.coords)[0][1]:
            min_idx -= 1
        if min_idx != 0 or tuple(ring[0]) != tuple(ring[-1]):
            ring = np.concatenate((ring[min_idx + 1:], ring[:min_idx + 1]), axis=0)
        return Polygon(ring)

    def placeholder_linepoly_baseline(self, region_polygon, img):
        """ Create a line-polygon (region-polygon scaled down by a factor of 0.75) and a baseline outside of the hull"""
        line_polygon = scale(region_polygon, xfact=0.75, yfact=0.75,
                             origin='centroid').convex_hull
        baseline = [((self.padding[0], img.img.shape[1]), (img.img.shape[0] - self.padding[1], img.img.shape[1]))]
        return line_polygon, baseline

    # TODO: Rewrite as soon as PAGEpy is available
    def predict(self):
        """ Main function to predict the line-polygons, baselines and texts in textregions """
        if not self.image:
            print(f"Missing image to process {self.xml}")
            return
        data = self.get_element_data()
        img_orig = ProcessedImage(self.image.get_filename(), background=self.background, orientation=0.0)
        if self.engine == 'tesseract':
            try:
                from tesserocr import PyTessBaseAPI, RIL, iterate_level
            except ModuleNotFoundError as err:
                # Error handling
                print(f'ModuleNotFoundError: tesserocr\n{err}')
                return
            with PyTessBaseAPI(lang=self.lang) as api:
                for element_idx, (element_id, entry) in enumerate(data.items()):
                    img = deepcopy(img_orig)
                    img.orientation = entry["orientation"]
                    img.cutout(shape=entry["coords"], padding=self.padding, background=self.background)
                    if self.deskew:
                        img.deskew(self.deskew)
                    elif self.auto_deskew:
                        img.auto_deskew()

                    region_polygon = MultiPoint(entry["coords"] -
                                                ([entry["coords"].min(axis=0)] - np.array([self.padding[0],
                                                                                           self.padding[
                                                                                               2]]))).convex_hull
                    if img.rotated_by:
                        region_polygon = rotate(region_polygon, img.rotated_by, 'center')
                    line_polygon = Polygon()

                    api.SetImage(fromarray(img.img))

                    psms = self.psm if entry['type'] != 'TextLine' else (13, 7)

                    for psm in psms:
                        api.SetPageSegMode(psm)
                        api.Recognize()
                        ri = api.GetIterator()
                        symbol_bboxs, symbol_baselines = self.tesseract_symbol_bboxs_and_baselines(ri,
                                                                                                   RIL, iterate_level)
                        if any(symbol_bboxs):
                            for idx, symbol_bbox in enumerate(symbol_bboxs):
                                if not region_polygon.covers(Point(symbol_bbox[0][:2])) and \
                                        not region_polygon.covers(Point(symbol_bbox[-1][2:])):
                                    line_polygon, symbol_baselines[idx] = self.placeholder_linepoly_baseline(
                                        region_polygon, img)
                            break
                    else:
                        if self.textline_placeholder:
                            line_polygon, symbol_baselines[0] = self.placeholder_linepoly_baseline(region_polygon, img)

                    line_index = 0
                    fulltext = ''

                    for lidx, line in enumerate(iterate_level(ri, RIL.TEXTLINE)):
                        print(
                            f"{self.xml.with_suffix('').name} "
                            f"({element_idx + 1}/{len(data)}):\t{element_id}l{line_index}")
                        if not line.Empty(RIL.TEXTLINE) or not line_polygon.is_empty:
                            """ XML - Output example:
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
                            linetext = self.text_wildcard
                            baseline = []
                            if not line.Empty(RIL.TEXTLINE):
                                baseline = line.Baseline(RIL.TEXTLINE)
                                linetext = self.text_wildcard if line.GetUTF8Text(RIL.TEXTLINE).strip() == '' \
                                    else line.GetUTF8Text(RIL.TEXTLINE).strip()
                                # Skip unrecognized textlines if set and or is last in the textblock in the block end
                                if self.delete_empty_textlines or \
                                        linetext == self.text_wildcard and lidx != 0 and lidx == len(symbol_bboxs) - 1:
                                    continue
                            if line_polygon.is_empty:
                                # The basic bbox list(line.BoundingBox(RIL.TEXTLINE)) is currently not used
                                # Experimental find bbox which are to wide (use only for horizontal text)
                                bboxs = self.shrink_bboxs(symbol_bboxs[lidx])
                                # Add missing points
                                line_polygon = self.polygon_from_bboxs(bboxs)
                                line_polygon = self.fit_line_to_region_polygon(line_polygon, region_polygon)
                                if line_polygon.is_empty or not isinstance(line_polygon, Polygon) or \
                                        line_polygon.length < 4 or (line_polygon.area / line_polygon.length) < 4:
                                    line_polygon = scale(region_polygon,
                                                         xfact=0.75, yfact=0.75, origin='centroid').convex_hull
                            # Fitting the baseline into the polygon
                            line_polygon = self.rotate_polygon_to_start_from_left_top(line_polygon)
                            baseline_linestring = self.fit_baseline2polygon(symbol_baselines[lidx],
                                                                            baseline, line_polygon)

                            # Undo rotation by deskewing
                            # Rotate degrees CCW from origin at the center of bbox
                            if img.rotated_by != 0:
                                pass
                                line_polygon = rotate(line_polygon, img.rotated_by*-1, 'center')
                                # Use the center of the line to rotate the baseline
                                baseline_linestring = rotate(baseline_linestring, img.rotated_by*-1,
                                                             interpret_origin(line_polygon, 'center', 2))

                            # Add offset to polygons/lines
                            line_polygon = translate(line_polygon, xoff=offset[0] - self.padding[0],
                                                     yoff=offset[1] - self.padding[2])
                            baseline_linestring = translate(baseline_linestring, xoff=offset[0] - self.padding[0],
                                                            yoff=offset[1] - self.padding[2])


                            if entry['type'] != 'TextLine':
                                line_attribs = {'id': f"{element_id}l{line_index}",
                                                'custom': f"readingOrder {{index:{line_index};}}"}
                                ele_textline = etree.SubElement(entry['element'], 'TextLine', line_attribs)
                                etree.SubElement(ele_textline, 'Coords',
                                             {'points': self.coords2str(list(line_polygon.exterior.coords))})
                                etree.SubElement(ele_textline, 'Baseline',
                                             {'points': self.coords2str(list(baseline_linestring.coords))})
                            # Writing TextEquiv parameters but index and conf aren't added currently
                            # not included: {'index': str(self.pred_index), 'conf': str(line.Confidence(RIL.TEXTLINE))})
                            else:
                                ele_textline = entry['element']
                            ele_textequiv = etree.SubElement(ele_textline, 'TextEquiv', self.text_equiv_attrib)
                            ele_unicode = etree.SubElement(ele_textequiv, 'Unicode')
                            ele_unicode.text = linetext
                            fulltext += '\n'+ele_unicode.text
                            line_index += 1
                            line_polygon = Polygon()
                        else:
                            print("No text found!")
                    # Does only TextRegion need a fulltext summary of TextLines?
                    if entry['element'].tag.endswith('TextLine'):
                        ele_parent = entry['element'].getparent()
                        if ele_parent.tag.endswith('TextRegion'):
                            ele_textequiv = ele_parent.find(f"./page:TextEquiv[@index='{self.pred_index}']",
                                                         namespaces=self.page.ns)
                            if ele_textequiv is None:
                                ele_textequiv = etree.SubElement(ele_parent,
                                                                 f"{{{self.page.ns['page']}}}TextEquiv",
                                                                 self.text_equiv_attrib)
                                ele_unicode = etree.SubElement(ele_textequiv, 'Unicode')
                                ele_unicode.text = fulltext.lstrip()
                            else:
                                ele_unicode = ele_textequiv.find('Unicode')
                                ele_unicode.text = (ele_unicode.text + fulltext).lstrip()
                    elif fulltext != '' and entry['element'].tag.endswith('TextRegion'):
                        ele_textregion = etree.SubElement(entry['element'], 'TextEquiv',
                                                          self.text_equiv_attrib)
                        ele_unicode = etree.SubElement(ele_textregion, 'Unicode')
                        ele_unicode.text = fulltext.lstrip()

    def export(self, output: Path):
        self.page.export(output)
