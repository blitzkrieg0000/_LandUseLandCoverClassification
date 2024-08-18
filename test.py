import re



band_list = [
    "/home/blitzkrieg/Downloads/spring/grid1/32UNV_20180407T102019_49_069550_10_141635/32UNV_20180407T102019_49_069550_10_141635_20m.tif",
    "/home/blitzkrieg/Downloads/spring/grid1/32UNV_20180407T102019_49_069550_10_141635/32UNV_20180407T102019_49_069550_10_141635_10m_IR.tif",
    "/home/blitzkrieg/Downloads/spring/grid1/32UNV_20180407T102019_49_069550_10_141635/32UNV_20180407T102019_49_069550_10_141635_10m_RGB.tif",
]

def SortByPatterns(path_list, data_filter):
    def match_priority(path):
        for i, pattern in enumerate(data_filter):
            if re.search(pattern, path):
                return i
        return len(data_filter)
    
    return sorted(path_list, key=match_priority)



results = SortByPatterns(band_list, [".*_10m_RGB", ".*_10m_IR", ".*_20m"])

print(results)