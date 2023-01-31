from datasets import Dataset as Ds
from datasets import DatasetDict as DsD
from datasets import load_from_disk
from typing import Union, Tuple


class Dataset(Ds):
    def train_val_split(self, 
                        val_frac: float =0.1,
                        return_dataset_dict: bool =True) -> Union[Tuple["Dataset", "Dataset"], "DatasetDict"]:
        """split dataset into tarin and validation datasets
        Args:
            dataset (Dataset): dataset to split
            val_frac (float, optional): validation radio of all dataset. Defaults to 0.1.
            return_dataset_dict (bool, optional): if return_dataset_dict is True, return a DatasetDict,
                otherwise return a tuple of train, val datasets. Defaults to True.

        Returns:
            Union[Tuple[Dataset, Dataset], DatasetDict]: if return_dataset_dict is True, return a DatasetDict, otherwise return a tuple of train, val datasets
        """
        df = self.to_pandas()
        train_df = df.sample(frac=1-val_frac)
        val_df = df.drop(train_df.index)
        train_ds = self.from_pandas(train_df, preserve_index=False)
        val_ds = self.from_pandas(val_df, preserve_index=False)
        if not return_dataset_dict:
            return train_ds, val_ds
        else:
            return DatasetDict({'train': train_ds, 'validation': val_ds})
        
    
    def train_val_test_split(self,
                             val_frac: float =0.1,
                             test_frac: float =0.1,
                             return_dataset_dict: bool =True) -> Union[Tuple["Dataset", "Dataset", "Dataset"], "DatasetDict"]:
        """split dataset into tarin vlidation and test datasets

        Args:
            dataset (Dataset): dataset to split
            val_frac (float, optional): validation radio of all dataset. Defaults to 0.1.
            test_frac (float, optional): test radio of all dataset. Defaults to 0.1.

        Returns:
            Union[Tuple[Dataset, Dataset, Dataset], DatasetDict]: if return_dataset_dict is True, return a DatasetDict, 
                otherwise return a tuple of train, val, test datasets
            
        """
        df = self.to_pandas()
        train_df = df.sample(frac=1-val_frac-test_frac)
        other_df = df.drop(train_df.index)
        val_df = other_df.sample(frac=1-(test_frac/(val_frac+test_frac)))
        test_df = other_df.drop(val_df.index)
        train_ds = self.from_pandas(train_df, preserve_index=False)
        val_ds = self.from_pandas(val_df, preserve_index=False)
        test_ds = self.from_pandas(test_df, preserve_index=False)
        if not return_dataset_dict:
            return train_ds, val_ds, test_ds
        else:
            return DatasetDict({'train': train_ds, 'validation': val_ds, 'test': test_ds})
        
class DatasetDict(DsD):
    @staticmethod
    def load_from_disk(dataset_path: str) -> "DatasetDict":
        return load_from_disk(dataset_path=dataset_path)