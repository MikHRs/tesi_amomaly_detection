import tempfile
import unittest
from pathlib import Path

from src.cvat_to_boxes import convert_cvat_xml
from src.inspect_cvat_export import inspect_xml


CVAT_XML = """\
<annotations>
  <meta>
    <task>
      <name>pilot</name>
      <source>pilot.mp4</source>
      <labels><label><name>anomalia</name></label></labels>
    </task>
  </meta>
  <track id="4" label="anomalia">
    <box frame="0" outside="0" occluded="0"
         xtl="10" ytl="20" xbr="40" ybr="70">
      <attribute name="tipo">persona</attribute>
      <attribute name="visibilita">chiaro</attribute>
      <attribute name="posizione">sui_binari</attribute>
    </box>
    <box frame="1" outside="1" occluded="0"
         xtl="11" ytl="20" xbr="41" ybr="70"/>
  </track>
  <image id="5" name="frame_test.jpg" width="100" height="80">
    <box label="anomalia" occluded="1"
         xtl="1" ytl="2" xbr="20" ybr="30"/>
    <polygon label="anomalia" points="1,2;3,4;5,6"/>
  </image>
</annotations>
"""


class CvatTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.xml_path = Path(self.temporary.name) / "annotations.xml"
        self.xml_path.write_text(CVAT_XML, encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_inspection(self) -> None:
        summary = inspect_xml(self.xml_path)
        self.assertEqual(summary["images"], 1)
        self.assertEqual(summary["tracks"], 1)
        self.assertEqual(summary["boxes"], 3)
        self.assertEqual(summary["polygons"], 1)
        self.assertEqual(summary["annotation_label_counts"], {"anomalia": 4})

    def test_conversion(self) -> None:
        rows = convert_cvat_xml(self.xml_path)
        self.assertEqual(len(rows), 3)
        active_track = next(row for row in rows if row["track_id"] == "4")
        self.assertEqual(active_track["video_or_image"], "pilot.mp4")
        self.assertEqual(active_track["label"], "anomalia")
        self.assertEqual(active_track["tipo"], "persona")
        self.assertEqual(active_track["visibilita"], "chiaro")
        self.assertEqual(active_track["posizione"], "sui_binari")


if __name__ == "__main__":
    unittest.main()
