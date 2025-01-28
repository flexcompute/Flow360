"""
Download volume, surface,acoustic etc... files to the directory it is run from.
"""

import os
import tarfile
import timeit

import click

import flow360 as fl

case_id = "ENTER CASE ID HERE"

# Used only if we are interested in downloading specific data sets
download_surfaces = True
download_volumes = True


case = fl.Case.from_cloud(case_id)
destination = os.path.join(os.getcwd(), case.name)

results = case.results

if os.path.exists(destination):
    overwrite_bool = click.confirm(
        f"Directory '{destination}' already exists, downloading might overwrite some of its content, do you want to continue?",
        default=True,
        abort=True,
    )
print('Beginning downloading')
results.download(all=True, destination=destination, overwrite=True)  # download all files generated

# Download only specific data sets
# results.download(
#     surface=download_surfaces,
#     volume=download_volumes,
#     destination=destination,
#     overwrite=True
# )

# Extract tar.gz files
tar_gz_files = [f for f in os.listdir(destination) if f.endswith(".tar.gz")]
for tar_gz_file in tar_gz_files:
    start = timeit.default_timer()
    file_path = os.path.join(destination, tar_gz_file)
    print(f"Processing: {file_path}")

    with tarfile.open(file_path, "r:gz") as tar:
        result_name = tar_gz_file[:-7]
        tar.extractall(path=os.path.join(destination, result_name))

    os.remove(file_path)
    print(f"  Removed: {file_path}")
    print(f"Extracting files for {tar_gz_file} done")
print('Downloading successful')
