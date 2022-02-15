from pagetools.src.Page import Page
from pagetools.src.Image import Image, ProcessedImage
from pagetools.src.utils import filesystem
from pagetools.src.utils.constants import extractable_regions

from pathlib import Path
from typing import List, Iterator, Set, Tuple
from collections import defaultdict
import re


class Extractor:
    def __init__(self, xml: Path, images: List[Path], include: List[str], exclude: List[str], no_text: bool, out: Path,
                 enumerate_output: bool, background, padding: Tuple[int], auto_deskew: bool, deskew: float,
                 gt_index: int, pred_index: int):
        self.xml = self.xml_to_page(xml)
        self.images = self.get_images(images)

        self.element_list = self.build_element_list(include, exclude)

        self.extract_text = not no_text
        self.extract_fulltext_only = True

        self.out = out
        self.enumerate_output = enumerate_output

        self.background = background
        self.padding = padding

        self.auto_deskew = auto_deskew
        self.deskew = deskew

        self.gt_index = gt_index
        self.pred_index = pred_index

    @staticmethod
    def xml_to_page(xml: Path):
        return Page(xml)

    @staticmethod
    def get_images(images: List[Path]) -> Iterator[Image]:
        return [Image(img) for img in images]

    @staticmethod
    def build_element_list(include: List[str], exclude: List[str]) -> Set[str]:
        element_list = extractable_regions.copy()
        if "*" in exclude:
            element_list.clear()
        elif exclude:
            for element_type in exclude:
                element_list.remove(element_type)
        if "*" in include:
            element_list = extractable_regions.copy()
        elif include:
            element_list.extend([elem_type for elem_type in include if elem_type != "*"])
        return element_list

    def reading_order(self):
        ro_dict = defaultdict()
        ro_ele = self.xml.tree.getroot().find(f".//page:ReadingOrder", namespaces=self.xml.ns)
        ordered_ro_ele = ro_ele.find(f".//page:OrderedGroup", namespaces=self.xml.ns)
        for child in ordered_ro_ele:
            ro_dict[child.get('regionRef')] = int(child.get('index', '0'))
        return ro_dict

    def fulltext(self):
        fulltexts = defaultdict(lambda: defaultdict(list))
        ro_dict = self.reading_order()
        for entry in self.xml.get_element_data({'TextRegion', 'TableRegion'}):
            if entry['type'] == 'TableRegion':
                ro = re.match('readingOrder {index:([0-9]*);}', entry['element'].get('custom', ''))
                ro = int(ro[1]) if ro else ro_dict.get(entry['element'].get('id'), -1)
                tablecontents = defaultdict(lambda: defaultdict(dict))
                for cell in entry['element'].findall("./page:TableCell", namespaces=self.xml.ns):
                    text_equivs = cell.findall('.//page:TextEquiv', namespaces=self.xml.ns)
                    content = defaultdict(str)
                    if text_equivs is not None:
                        for text_equiv in text_equivs:
                            content[text_equiv.get('index')] += ' ' + \
                                                                "".join(text_equiv.find(
                                                                    "./page:Unicode",
                                                                    namespaces=self.xml.ns).itertext())
                    else:
                        content['all'] += ''
                    max_row = int(cell.get('row')) + int(cell.get('rowSpan')) - 1
                    max_col = int(cell.get('col')) + int(cell.get('colSpan')) - 1
                    for row in range(int(cell.get('row')), max_row+1):
                        for col in range(int(cell.get('col')), max_col+1):
                            for table_index, c_text in content.items():
                                if row != max_row or col != int(cell.get('col')):
                                    tablecontents[table_index][row][col] = ''
                                else:
                                    tablecontents[table_index][row][col] = c_text.lstrip()
                for all_contents in tablecontents.pop('all', []):
                    for table_index in tablecontents.keys():
                        tablecontents[table_index].update(all_contents)
                for index, tablecontent in tablecontents.items():
                    fulltexts[index][ro].extend(['\t'.join([col_content for col, col_content in sorted(col_dict.items())])
                                                 for row, col_dict in sorted(tablecontent.items())])

            for text_equiv in entry['text_equivs']:
                ro = re.match('readingOrder {index:([0-9]*);}', entry['element'].get('custom', ''))
                ro = int(ro[1]) if ro else ro_dict.get(entry['element'].get('id'), -1)
                fulltexts[text_equiv['index']][ro].append(text_equiv['content'])
        for index, fulltext in fulltexts.items():
            with self.xml.get_filename().with_suffix(f"{'' if index is None else '_'+index if index > 0 else '.gt'}"
                                                     f".txt").open("w") as textfile:
                textfile.write('\n'.join(['\n'.join(text_content) for ro, text_content in sorted(fulltext.items())]))
        return

    # TODO: Rewrite as soon as PAGEpy is available
    def extract(self, enumerator):

        if self.extract_fulltext_only:
            return self.fulltext()

        data = self.xml.get_element_data(self.element_list)

        for image in self.images:
            for entry in data:
                img = ProcessedImage(image.get_filename(), background=self.background, orientation=entry["orientation"])

                img.cutout(shape=entry["coords"], padding=self.padding, background=self.background)

                if self.deskew:
                    img.deskew(self.deskew)
                elif self.auto_deskew:
                    img.auto_deskew()

                img_suffix = filesystem.get_suffix(image.get_filename())
                file_basename = filesystem.get_file_basename(self.xml.get_filename())

                if self.enumerate_output:
                    base_filename = Path(self.out, f"{str(enumerator[0]).zfill(6)}")
                else:
                    base_filename = Path(self.out, f"{file_basename}_{entry['id']}")

                img_filename = base_filename.with_suffix(img_suffix)
                img.export_image(img_filename)

                if self.extract_text:
                    num_non_indexed_text_equivs = 0
                    for text_equiv in entry["text_equivs"]:
                        if text_equiv["index"] is None:
                            text_suffix = f".u{num_non_indexed_text_equivs}.txt"
                            num_non_indexed_text_equivs += 1
                        elif int(text_equiv["index"]) == self.gt_index:
                            text_suffix = ".gt.txt"
                        elif int(text_equiv["index"]) == self.pred_index:
                            text_suffix = ".pred.txt"
                        else:
                            text_suffix = f".i{text_equiv['index']}.txt"
                        filesystem.write_text_file(text_equiv["content"], base_filename.with_suffix(text_suffix))

                enumerator[0] += 1
