from readers.DataAndNumEntriesReader import DataAndNumEntriesReader
from readers.FixedNumEntriesReader import FixNumEntriesReader,FixNumEntriesReaderSeeds, FixNumEntriesReaderSeedsSeparate


class ReaderFactory:
    def get_class(self, reader_name):
        if reader_name == "fixed_num_entries_reader":
            return FixNumEntriesReader
        if reader_name == "fixed_num_entries_reader_seeds":
            return FixNumEntriesReaderSeeds
        elif reader_name == "data_and_num_entries_reader":
            return DataAndNumEntriesReader
        if reader_name == "fixed_num_entries_reader_seeds_separate":
            return FixNumEntriesReaderSeedsSeparate