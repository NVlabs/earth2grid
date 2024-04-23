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
