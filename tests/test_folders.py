from flow360 import Case, Folder
from flow360.log import set_logging_level

set_logging_level("DEBUG")


def test_case(mock_id, mock_response: None):
    # create folder in ROOT level
    folder_A = Folder.create("folder-python-level-A").submit()
    print(folder_A)

    # create folder inside the above folder
    folder_B = Folder.create("folder-python-level-B", parent_folder=folder_A).submit()

    assert folder_B.info.parent_folder_id == folder_A.id
    print(folder_B)

    # create folder in ROOT level and move inside folder_B
    folder_C = Folder.create("folder-python-level-C").submit()
    folder_C = folder_C.move_to_folder(folder_B)
    print(folder_C)

    Case(id=mock_id).move_to_folder(folder_C)
