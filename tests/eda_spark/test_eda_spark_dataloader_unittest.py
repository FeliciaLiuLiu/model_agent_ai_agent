import sqlite3
import sys
import unittest
from pathlib import Path

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from _common import (  # noqa: E402
    SparkTestCase,
    write_csv,
    write_json_array,
    write_nb_df_loader,
    write_nb_loader,
    write_parquet_with_spark,
    write_py_df_loader,
    write_py_loader,
    write_sql_file,
    write_sqlite_db,
    write_tsv,
)
from eda_spark import dataloader as dl  # noqa: E402
from eda_spark.dataloader import DataLoader  # noqa: E402


class TestSparkDataLoaderHelpers(SparkTestCase):
    def test_is_glob_and_as_list(self):
        self.assertTrue(dl._is_glob("*.csv"))
        self.assertFalse(dl._is_glob("data/input.csv"))
        self.assertEqual(dl._as_list(None), [])
        self.assertEqual(dl._as_list("x"), ["x"])
        self.assertEqual(dl._as_list(("a", "b")), ["a", "b"])

    def test_expand_paths_for_dir_glob_and_recursive(self):
        write_csv(self.tmp_path, name="a.csv", rows=2)
        write_tsv(self.tmp_path, name="b.tsv", rows=1)
        nested = self.tmp_path / "nested"
        nested.mkdir()
        write_csv(nested, name="c.csv", rows=1)

        direct = dl._expand_paths([str(self.tmp_path)], recursive=False)
        names = {p.name for p in direct}
        self.assertIn("a.csv", names)
        self.assertIn("b.tsv", names)
        self.assertNotIn("c.csv", names)

        globbed = dl._expand_paths([str(self.tmp_path / "*.csv")], recursive=False)
        self.assertEqual({p.name for p in globbed}, {"a.csv"})

        rec = dl._expand_paths([str(self.tmp_path)], recursive=True)
        self.assertIn("c.csv", {p.name for p in rec})

    def test_resolve_and_scan_dir(self):
        write_csv(self.tmp_path, name="scan.csv", rows=1)
        resolved = dl._resolve_data_dir(str(self.tmp_path))
        self.assertEqual(resolved, self.tmp_path)

        scanned = dl._scan_dir_for_exts(str(self.tmp_path), [".csv"], recursive=False)
        self.assertEqual(len(scanned), 1)
        self.assertEqual(scanned[0].name, "scan.csv")

    def test_read_excel_with_spark_returns_none_when_unavailable(self):
        result = dl._read_excel_with_spark(self.spark, "file:///definitely-not-exists.xlsx")
        self.assertIsNone(result)

    def test_read_single_spark_for_supported_formats(self):
        csv_path = write_csv(self.tmp_path, rows=3)
        tsv_path = write_tsv(self.tmp_path, rows=2)
        json_path = write_json_array(self.tmp_path, rows=4)
        parquet_path = write_parquet_with_spark(self.tmp_path, self.spark, rows=5)

        self.assertEqual(dl._read_single_spark(self.spark, csv_path).count(), 3)
        self.assertEqual(dl._read_single_spark(self.spark, tsv_path).count(), 2)
        self.assertEqual(dl._read_single_spark(self.spark, json_path).count(), 4)
        self.assertEqual(dl._read_single_spark(self.spark, parquet_path).count(), 5)

    def test_read_single_spark_unsupported_extension(self):
        txt = self.tmp_path / "bad.txt"
        txt.write_text("x", encoding="utf-8")
        with self.assertRaises(ValueError):
            dl._read_single_spark(self.spark, txt)

    def test_concat_frames(self):
        df1 = self.spark.createDataFrame(pd.DataFrame({"a": [1], "b": ["x"]}))
        df2 = self.spark.createDataFrame(pd.DataFrame({"a": [2], "b": ["y"]}))
        union_df = dl._concat_frames([df1, df2])
        self.assertEqual(union_df.count(), 2)
        with self.assertRaises(ValueError):
            dl._concat_frames([])

    def test_python_file_and_code_and_notebook_loaders(self):
        py_load = write_py_loader(self.tmp_path)
        py_df = write_py_df_loader(self.tmp_path)
        nb_load = write_nb_loader(self.tmp_path)
        nb_df = write_nb_df_loader(self.tmp_path)

        self.assertEqual(dl._load_python_file(str(py_load), self.spark).count(), 2)
        self.assertEqual(dl._load_python_file(str(py_df), self.spark).count(), 3)
        self.assertEqual(dl._load_notebook(str(nb_load), self.spark).count(), 3)
        self.assertEqual(dl._load_notebook(str(nb_df), self.spark).count(), 2)

        code_load = (
            "import pandas as pd\n"
            "def load():\n"
            "    return pd.DataFrame({'a':[1,2],'b':['x','y']})\n"
        )
        code_df = "import pandas as pd\ndf = pd.DataFrame({'a':[1],'b':['x']})\n"
        self.assertEqual(dl._load_python_code(code_load, self.spark).count(), 2)
        self.assertEqual(dl._load_python_code(code_df, self.spark).count(), 1)

    def test_python_notebook_loader_errors(self):
        with self.assertRaises(FileNotFoundError):
            dl._load_python_file(str(self.tmp_path / "missing.py"), self.spark)
        with self.assertRaises(ValueError):
            dl._load_python_code("x = 1", self.spark)

        bad_nb = self.tmp_path / "bad.ipynb"
        bad_nb.write_text('{"cells":[{"cell_type":"code","source":"x = 1"}]}', encoding="utf-8")
        with self.assertRaises(ValueError):
            dl._load_notebook(str(bad_nb), self.spark)

    def test_ensure_spark_df(self):
        spark_df = self.spark.createDataFrame(pd.DataFrame({"a": [1]}))
        pandas_df = pd.DataFrame({"a": [1, 2]})
        self.assertEqual(dl._ensure_spark_df(spark_df, self.spark).count(), 1)
        self.assertEqual(dl._ensure_spark_df(pandas_df, self.spark).count(), 2)
        with self.assertRaises(ValueError):
            dl._ensure_spark_df("not-a-df", self.spark)

    def test_sql_helpers(self):
        sql = """
        -- comment
        SELECT * FROM t
        """
        self.assertEqual(dl._clean_sql_lines(sql), ["SELECT * FROM t"])
        self.assertTrue(dl._is_select_only("SELECT * FROM t"))
        self.assertTrue(dl._is_select_only("WITH x AS (SELECT 1) SELECT * FROM x"))
        self.assertFalse(dl._is_select_only("SELECT 1; SELECT 2"))
        self.assertFalse(dl._is_select_only("CREATE TABLE t(a INT)"))

        db_path = write_sqlite_db(self.tmp_path)
        conn = sqlite3.connect(db_path)
        try:
            objects = dl._sqlite_objects(conn)
        finally:
            conn.close()
        self.assertIn(("t", "table"), objects)
        self.assertEqual(dl._pick_sql_table([("aml_dataset", "table"), ("x", "table")]), "aml_dataset")
        self.assertEqual(dl._pick_sql_table([("only_table", "table")]), "only_table")
        self.assertIsNone(dl._pick_sql_table([("a", "table"), ("b", "table")]))

    def test_load_sql_with_db(self):
        db_path = write_sqlite_db(self.tmp_path)
        df = dl._load_sql_with_db(self.spark, "SELECT * FROM t", f"sqlite:///{db_path}")
        self.assertEqual(df.count(), 2)
        df2 = dl._load_sql_with_db(self.spark, "SELECT * FROM t", str(db_path))
        self.assertEqual(df2.count(), 2)

    def test_load_sql_auto(self):
        db_path = write_sqlite_db(self.tmp_path)
        query = write_sql_file(self.tmp_path, name="query.sql", sql_text="SELECT * FROM t")
        frames = dl._load_sql_auto(self.spark, [query], [db_path])
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].count(), 2)

        script = write_sql_file(
            self.tmp_path,
            name="script.sql",
            sql_text=(
                "DROP TABLE IF EXISTS aml_dataset;\n"
                "CREATE TABLE aml_dataset (a INT, b TEXT);\n"
                "INSERT INTO aml_dataset VALUES (1, 'x');\n"
            ),
        )
        frames2 = dl._load_sql_auto(self.spark, [script], [])
        self.assertEqual(len(frames2), 1)
        self.assertEqual(frames2[0].count(), 1)

    def test_load_sql_auto_multiple_tables_error(self):
        script = write_sql_file(
            self.tmp_path,
            name="many.sql",
            sql_text=(
                "CREATE TABLE a (x INT);\n"
                "INSERT INTO a VALUES (1);\n"
                "CREATE TABLE b (y INT);\n"
                "INSERT INTO b VALUES (2);\n"
            ),
        )
        with self.assertRaises(ValueError):
            dl._load_sql_auto(self.spark, [script], [])


class TestSparkDataLoaderClass(SparkTestCase):
    def test_detect_mode(self):
        self.assertEqual(DataLoader(self.spark, data=["x.csv"])._detect_mode(), "data")
        self.assertEqual(DataLoader(self.spark, sql="SELECT 1")._detect_mode(), "sql")
        self.assertEqual(DataLoader(self.spark, py="x.py")._detect_mode(), "py")
        self.assertEqual(DataLoader(self.spark, py_code="x=1")._detect_mode(), "py_code")
        self.assertEqual(DataLoader(self.spark, nb="x.ipynb")._detect_mode(), "nb")
        self.assertEqual(DataLoader(self.spark, auto_exec=True)._detect_mode(), "auto")
        with self.assertRaises(ValueError):
            DataLoader(self.spark, data=["a.csv"], py="x.py")._detect_mode()

    def test_load_data_mode_with_explicit_path(self):
        csv_path = write_csv(self.tmp_path, rows=3)
        df, src = DataLoader(self.spark, data=[str(csv_path)]).load()
        self.assertEqual(df.count(), 3)
        self.assertIn("input.csv", src)

    def test_load_data_mode_with_directory_and_recursive(self):
        write_csv(self.tmp_path, name="part1.csv", rows=2)
        nested = self.tmp_path / "nested"
        nested.mkdir()
        write_csv(nested, name="part2.csv", rows=3)

        non_recursive_df, _ = DataLoader(self.spark, data=[str(self.tmp_path)], recursive=False).load()
        self.assertEqual(non_recursive_df.count(), 2)

        recursive_df, _ = DataLoader(self.spark, data=[str(self.tmp_path)], recursive=True).load()
        self.assertEqual(recursive_df.count(), 5)

    def test_load_data_mode_auto_latest(self):
        write_csv(self.tmp_path, name="auto.csv", rows=4)
        df, src = DataLoader(self.spark, data=None, data_dir=str(self.tmp_path)).load()
        self.assertEqual(df.count(), 4)
        self.assertIn("auto.csv", src)

    def test_load_sql_mode_requires_db(self):
        with self.assertRaises(ValueError):
            DataLoader(self.spark, sql="SELECT 1").load()

    def test_load_sql_mode(self):
        db_path = write_sqlite_db(self.tmp_path)
        df, src = DataLoader(self.spark, sql="SELECT * FROM t", db=f"sqlite:///{db_path}").load()
        self.assertEqual(df.count(), 2)
        self.assertTrue(src.startswith("sql:"))

    def test_load_py_mode(self):
        py_path = write_py_loader(self.tmp_path)
        df, src = DataLoader(self.spark, py=str(py_path)).load()
        self.assertEqual(df.count(), 2)
        self.assertTrue(src.startswith("py:"))

    def test_load_py_code_mode(self):
        code = "import pandas as pd\ndef load():\n    return pd.DataFrame({'a':[1,2,3]})\n"
        df, src = DataLoader(self.spark, py_code=code).load()
        self.assertEqual(df.count(), 3)
        self.assertEqual(src, "py_code")

    def test_load_notebook_mode(self):
        nb_path = write_nb_loader(self.tmp_path)
        df, src = DataLoader(self.spark, nb=str(nb_path)).load()
        self.assertEqual(df.count(), 3)
        self.assertTrue(src.startswith("nb:"))

    def test_load_auto_mode(self):
        write_csv(self.tmp_path, name="data.csv", rows=3)
        write_sqlite_db(self.tmp_path)
        write_sql_file(self.tmp_path, sql_text="SELECT * FROM t")
        write_py_loader(self.tmp_path)
        write_nb_loader(self.tmp_path)

        df, src = DataLoader(self.spark, data_dir=str(self.tmp_path), auto_exec=True).load()
        self.assertEqual(df.count(), 10)
        self.assertIn("data.csv", src)
        self.assertIn("query.sql", src)
        self.assertIn("loader.py", src)
        self.assertIn("loader.ipynb", src)

    def test_load_auto_mode_no_supported_file(self):
        with self.assertRaises(FileNotFoundError):
            DataLoader(self.spark, data_dir=str(self.tmp_path), auto_exec=True).load()

    def test_load_data_mode_no_match(self):
        with self.assertRaises(FileNotFoundError):
            DataLoader(self.spark, data=[str(self.tmp_path / "*.csv")]).load()


if __name__ == "__main__":
    unittest.main()
