# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A script to check that copyright headers exists"""

import json
import re
import subprocess
from datetime import datetime
from pathlib import Path


def get_top_comments(_data):
    """
    Get all lines where comments should exist
    """
    lines_to_extract = []
    for i, line in enumerate(_data):
        # If empty line, skip
        if line in ["", "\n", "", "\r", "\r\n"]:
            continue
        # If it is a comment line, we should get it
        if line.startswith("#"):
            lines_to_extract.append(i)
        # Assume all copyright headers occur before any import or from statements
        # and not enclosed in a comment block
        elif "import" in line:
            break
        elif "from" in line:
            break

    comments = [_data[line] for line in lines_to_extract]
    return comments


def main():

    with open(Path(__file__).parent.resolve() / Path("config.json")) as f:
        config = json.loads(f.read())
    print("License check config:")
    print(json.dumps(config, sort_keys=True, indent=4))

    current_year = int(datetime.today().year)
    starting_year = 2023
    python_header_path = Path(__file__).parent.resolve() / Path(config["copyright_file"])
    exts = config["include-ext"]

    with open(python_header_path, "r", encoding="utf-8") as original:
        pyheader = original.read().split("\n")
        pyheader_lines = len(pyheader)

    # Build list of files to check
    # has ext in include-ext but doesn't start with any of the exclude-dir
    git_files_output = subprocess.check_output(["git", "ls-files"])  # noqa: S603, S607
    filenames = [line.decode("utf-8").strip() for line in git_files_output.splitlines()]
    filenames = [filename for filename in filenames if any(filename.endswith(ext) for ext in exts)]
    filenames = [
        filename
        for filename in filenames
        if not any(str(filename).startswith(str(exclude)) for exclude in config["exclude-dir"])
    ]

    problematic_files = []
    gpl_files = []

    for filename in filenames:
        with open(str(filename), "r", encoding="utf-8") as original:
            data = original.readlines()

        data = get_top_comments(data)
        if data and "# ignore_header_test" in data[0]:
            continue
        if len(data) < pyheader_lines - 1:
            print(f"{filename} has less header lines than the copyright template")
            problematic_files.append(filename)
            continue

        found = False
        for i, line in enumerate(data):
            if re.search(re.compile("Copyright.*NVIDIA.*", re.IGNORECASE), line):
                found = True
                # Check 1st line manually
                year_good = False
                for year in range(starting_year, current_year + 1):
                    year_line = pyheader[0].format(CURRENT_YEAR=year)
                    if year_line in data[i]:
                        year_good = True
                        break
                    year_line_aff = year_line.split(".")
                    year_line_aff = year_line_aff[0] + " & AFFILIATES." + year_line_aff[1]
                    if year_line_aff in data[i]:
                        year_good = True
                        break
                if not year_good:
                    problematic_files.append(filename)
                    print(f"{filename} had an error with the year")
                    break
                # while "opyright" in data[i]:
                #    i += 1
                # for j in range(1, pyheader_lines):
                #    if pyheader[j] not in data[i + j - 1]:
                #        problematic_files.append(filename)
                #        print(f"{filename} missed the line: {pyheader[j]}")
                #        break
            if found:
                break
        if not found:
            print(f"{filename} did not match the regex: `Copyright.*NVIDIA.*`")
            problematic_files.append(filename)

        # test if GPL license exists
        for lines in data:
            if "gpl" in lines.lower():
                gpl_files.append(filename)
                break

    if len(problematic_files) > 0:
        print("The following files that might not have a copyright header:")
        for _file in problematic_files:
            print(_file)
    if len(gpl_files) > 0:
        print("test_header.py found the following files that might have GPL copyright:")
        for _file in gpl_files:
            print(_file)
    assert len(problematic_files) == 0, "header test failed!"
    assert len(gpl_files) == 0, "found gpl license, header test failed!"

    print(f"Success: File headers of {len(filenames)} files look good!")


if __name__ == "__main__":
    main()
