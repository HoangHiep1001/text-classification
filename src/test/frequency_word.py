import re
from collections import Counter



if __name__ == '__main__':
    s = ['dien_anh', 'du_lich', 'giao_duc', 'kinh_doanh', 'ngan_hang', 'suc_khoe', 'the_thao',
         'thoi_su_phap_luat']
    for str1 in s:
        words = re.findall(r'\w+', open('../../data/data_process/'+str1+'/'+str1+'.txt').read().lower())
        print(Counter(words).most_common(20))