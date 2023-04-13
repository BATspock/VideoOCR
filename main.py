from getText import find_text
import os
import timeit
# get all videos in the Videos folder

video = os.listdir("Videos")

for videos in video:
    time_start = timeit.default_timer()
    find_text("Videos/"+videos)
    with open("time.txt", "a") as f:
        f.write("Time taken {}: {} seconds\n".format(videos, timeit.default_timer() - time_start))

    f.close()


