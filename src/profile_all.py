import cProfile
import pstats
from pstats import SortKey

from detectvehicles import run

create_stats = True
read_stats = True
file_name = "profile_results"

def readStats():
    p = pstats.Stats(file_name)
    p.sort_stats(SortKey.TIME, SortKey.PCALLS).print_stats(15)

if __name__=="__main__":
    if create_stats:
        cProfile.run('run(source="../resources/videos/production ID 4979730.mp4", read_inputs_from_csv=True, nosave=True, view_img=False)', file_name)

    if read_stats:
        readStats()