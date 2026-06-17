import lxml.etree
import sarkit.sicd as sksicd

try:
    from smart_open import open
except ImportError:
    pass


def read_sicd_xml(filename):
    """Parse SICD XML from a SICD NITF or XML file"""
    with open(filename, "rb") as file:
        try:
            xmltree = lxml.etree.parse(file)
        except lxml.etree.XMLSyntaxError:
            file.seek(0)
            with sksicd.NitfReader(file) as reader:
                xmltree = reader.metadata.xmltree
    return xmltree
