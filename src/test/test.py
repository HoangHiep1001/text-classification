import codecs
import regex as re
import pandas as pd
from gensim.models import KeyedVectors, Word2Vec


def remove_stopword(text):
    filename = '../../data/stopwords.txt'
    list_stopwords = []
    with codecs.open(filename, "r", encoding="utf8") as file:
        a = file.readlines()
        for i in a:
            list_stopwords.append(i.strip())
    pre_text = []
    words = text.split()
    for word in words:
        if word not in list_stopwords:
            pre_text.append(word)
        text2 = ' '.join(pre_text)
    return text2

if __name__ == '__main__':
    text = "nguyễn đức bảo lớp trường tiểu_học lâm thanh chương nghệ_an toán_học mẫu_giáo học giải toán lớp thú_vị thơ nguyễn đức bảo học toán thơ toán lâm thanh chương nghệ_an thăm đường nguyễn đức bảo dân nhiệt tỉnh tôi quên thêm thằng bảo giải toán giỏi học lớp giải toán lớp dân bảo xóm minh sơn chuẩn giao_lưu toán thơ toàn_quốc bảo bố_mẹ ngoại kia sông xả_hơi nguyễn tùng bảo giấu niềm hào trai mình người_ta đồn bảo thể giải toán thực_tế cháu bắt_đầu giải toán lớp thôi tất cháu học giải nguyễn văn lợng tuổi nội bảo đầu_tiên phát_hiện khả_năng toán cháu đích_tôn hồi bảo tuổi lớp mẫu_giáo nhỡ mải quan_tâm múa hát đặc_biệt tập đếm phép cộng_trừ ngày học tối hai cháu ngủ nhau hắn bắt tui phép giải_đáp phép cộng trừ phạm_vi hầu hắn hết lợng biết học toán năng_khiếu môn_học này tùng mua sách toán lớp trai toán sách toán lớp mua sách toán lớp luyện tùng hay tôi vốn giáo_viên dạy toán quan cháu phát_triển nhiên khả_năng cháu tham_gia dạy_học cháu học toán thỉnh_thoảng quá cháu giúp_đỡ hướng thôi học lớp giải toán lớp tầm rưỡi bảo kiểm_tra khả_năng toán quyển sách toán lớp bảo trực_tiếp tập đôi mắt tròn thông mình làn ngăm đen cười rạng_rỡ mặc_cả giải toán lớp tập bất_kỳ sách giáo_khoa lớp đề_nghị bảo giải phút chàng xách bàn kiểm_tra lại giờ cháu giải toán lớp học đầu thôi toán lớp cháu giải xong rồi cháu học hình đại hình thú_vị hơn bảo tâm_sự lớp cháu chút khả_năng toán_học cháu học cháu thể phát_huy khả_năng gửi cháu học lớp năng_khiếu gia_đình kiện hùng biết thời_gian học giải toán lớp trên tất tập giải_quyết lớp xong bố_mẹ mua sách cho cháu đọc lý_thuyết áp_dụng giải cháu giải hướng sách giáo_khoa cháu giải mình hầu cháu giải tất_tập thỉnh_thoảng cháu thôi cháu trở_thành toán_học cháu hâm_mộ ngô bảo châu mong trở_thành toán_học bảo say_sưa tâm ước_mơ mình giải thi giải toán mạng internet khối học_sinh lớp toàn tỉnh nghệ_an tập_trung ôn_luyện tham_dự thi toàn_quốc tổ_chức đầu tới cháu hy_vọng giảnh giải hội ngô bảo châu tác thơ năng_khiếu toán_học nguyễn đức bảo học giỏi môn tiếng anh văn_thơ văn đọc văn bảo dòng viết học_trò tuổi dòng chữ_viết ngắn tròn_trịa văn người_lớn diễn xung_quanh bảo toán khó bảo giúp_đỡ bất_ngờ đọc thơ học_trò học toán tác thơ nỗi quan_sát cảm sống quanh mình mắt buồn rười_rượi môi nhạt tím dần gầy đau_nhói thương vụng nhặt rơi trích thơ cháu viết đội chiến_đấu thăm chiến_trường nhà bảo lý_giải cảnh tết thơ bảo sức thân_thuộc gũi những cánh hoa đào hương tết chợ tết rộn_ràng mơ_màng xưa bao_nhiêu thơ xe_đạp lơ_thơ mơ_màng trông dân thụt nấu bánh trích cảnh tết hồi cháu thơ đọc thơ cháu bảo cháu phải cháu đấy cháu viết cháu viết cháu thơ nữa bảo thêm học_trò đặc_biệt mình thầy nguyễn văn quân chủ_nhiệm lớp trường tiểu_học lâm thanh chương biết nguyễn đức bảo đặc_biệt khă năng môn toán khả nhẩm trí tốt tinh_thần_học cao tòi khám_phá học môn đặc_biệt nổi_trội môn toán bồi_dưỡng thể tiến hơn hoàng lam"
    print(len(text))