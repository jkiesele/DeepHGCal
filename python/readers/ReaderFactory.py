from readers.DataAndNumEntriesReader import DataAndNumEntriesReader
from readers.FixedNumEntriesReader import FixNumEntriesReader


class ReaderFactory:
    def get_class(self, reader_name):
        if reader_name == "fixed_num_entries_reader":
            return FixNumEntriesReader
        elif reader_name == "data_and_num_entries_reader":
            return DataAndNumEntriesReader