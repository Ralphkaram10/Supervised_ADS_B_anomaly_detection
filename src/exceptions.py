class WrongConfigKeysError(Exception):
    """Exception raised when there is at least one wrong key in the config dictionary obtained from the yaml config file.
    """
    def __init__(self, required_config_keys,actual_config_keys):
        self.message=""
        if len(required_config_keys-actual_config_keys)>0:
            self.message+=f"\nThe following keys need to be added to the config: {required_config_keys-actual_config_keys}"
        if len(actual_config_keys-required_config_keys)>0:
            self.message+=f"\nThe following keys need to be removed from the config: {actual_config_keys-required_config_keys}"
        super().__init__(self.message)

    def __str__(self):
        return self.message

class DatasetNotFoundError(Exception):
    """Exception raised if dataset is not found in "../datasets" directory

    """
    def __init__(self,dataset_name):
        self.dataset_name = dataset_name
        self.message=f"\nDirectory \"{dataset_name}\" not found in \"../datasets\" directory or is empty"
        super().__init__(self.message)
        
def verify_strictly_positive(i,str="lookback"):
    if not isinstance(i,int):
        raise TypeError(f"{str} should be a strictly positive int")
    if i<=0:
        raise ValueError(f"{str} should be a strictly positive int")

def verify_if_in_list(i,str="",list=[]):
    if i not in list:
        raise ValueError(f"{str} should be in {list}")

        
        
