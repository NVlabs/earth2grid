# Copyright (c) 2015, Óscar Nájera
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of sphinx-gallery nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# from https://sphinx-gallery.github.io/stable/advanced.html#example-2-detecting-image-files-on-disk
import os
import shutil
from glob import glob

from sphinx_gallery.scrapers import figure_rst


class PNGScraper(object):
    def __init__(self):
        self.seen = set()

    def __repr__(self):
        return 'PNGScraper'

    def __call__(self, block, block_vars, gallery_conf):
        # Find all PNG files in the directory of this example.
        path_current_example = os.path.dirname(block_vars['src_file'])
        pngs = sorted(glob(os.path.join(path_current_example, '*.png')))

        # Iterate through PNGs, copy them to the sphinx-gallery output directory
        image_names = list()
        image_path_iterator = block_vars['image_path_iterator']
        for png in pngs:
            if png not in self.seen:
                self.seen |= set(png)
                this_image_path = image_path_iterator.next()
                image_names.append(this_image_path)
                shutil.move(png, this_image_path)
        # Use the `figure_rst` helper function to generate reST for image files
        return figure_rst(image_names, gallery_conf['src_dir'])
