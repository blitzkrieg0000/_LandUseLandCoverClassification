import re

def SortByPatterns(path_list, data_filter):
    def match_priority(path):
        for i, pattern in enumerate(data_filter):
            if re.search(pattern, path):
                return i
        return len(data_filter)
    
    return sorted(path_list, key=match_priority)

# Örnek path listesi
path_list = [
    "31UGR_20180418T104021_50_007153_6_167685_10m_RGB.tif",
    "31UGR_20180418T104021_50_007153_6_167685_20m_RGB.tif",
    "31UGR_20180418T104021_50_007153_6_167685_IR.tif",
    "31UGR_20180418T104021_50_007153_6_167685_20m_IR.tif",
    "31UGR_20180418T104021_50_007153_6_167685_10m_IR.tif",
]

data_filter = [".*_10m", ".*_20m", ".*_IR"]

sorted_paths = sort_by_patterns(path_list, data_filter)
print("\n".join(sorted_paths))
