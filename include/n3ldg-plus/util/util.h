#ifndef _MYLIB_H_
#include <fstream>
#define _MYLIB_H_

#include <functional>
#include <string>
#include <vector>
#include <fstream>
#include <cassert>
#include <deque>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <cmath>
#include <ctime>
#include <cfloat>
#include <cstring>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <memory>

#include "n3ldg-plus/util/nrmat.h"
#include "eigen/Eigen/Dense"
#include "n3ldg-plus/base/def.h"

namespace n3ldg_plus {

//typedef long long blong;

const static n3ldg_plus::dtype minlogvalue = -1000;
const static n3ldg_plus::dtype d_zero = 0.0;
const static n3ldg_plus::dtype d_one = 1.0;
const static std::string nullkey = "-NULL-";
const static std::string unknownkey = "-UNKNOWN-";
const static std::string seperateKey = "#";
//#ifdef _WIN32
//    "\\";
//#else
//    "/";
//#endif

//typedef std::vector<std::string> CStringVector;
//typedef std::vector<std::pair<std::string, std::string> > CTwoStringVector;

//class string_less {
//  public:
//    bool operator()(const std::string &str1, const std::string &str2) const {
//        int ret = strcmp(str1.c_str(), str2.c_str());
//        if (ret < 0)
//            return true;
//        else
//            return false;
//    }
//};

//class LabelScore {
//  public:
//    int labelId;
//    n3ldg_plus::dtype score;

//  public:
//    LabelScore() {
//        labelId = -1;
//        score = 0.0;
//    }
//    LabelScore(int id, n3ldg_plus::dtype value) {
//        labelId = id;
//        score = value;
//    }
//};

//class LabelScore_Compare {
//  public:
//    bool operator()(const LabelScore &o1, const LabelScore &o2) const {

//        if (o1.score < o2.score)
//            return -1;
//        else if (o1.score > o2.score)
//            return 1;
//        else
//            return 0;
//    }
//};

///*==============================================================
// *
// * CSentenceTemplate
// *
// *==============================================================*/

//template<typename CSentenceNode>
//class CSentenceTemplate : public std::vector<CSentenceNode> {

//  public:
//    CSentenceTemplate() {
//    }
//    virtual ~CSentenceTemplate() {
//    }
//};


//template<typename CSentenceNode>
//std::istream & operator >> (std::istream &is, CSentenceTemplate<CSentenceNode> &sent) {
//    sent.clear();
//    std::string line;
//    while (is && line.empty())
//        getline(is, line);

//    while (is && !line.empty()) {
//        CSentenceNode node;
//        std::istringstream iss(line);
//        iss >> node;
//        sent.push_back(node);
//        getline(is, line);
//    }
//    return is;
//}

//template<typename CSentenceNode>
//std::ostream & operator <<(std::ostream &os, const CSentenceTemplate<CSentenceNode> &sent) {
//    for (unsigned i = 0; i < sent.size(); ++i)
//        os << sent.at(i) << std::endl;
//    os << std::endl;
//    return os;
//}

//inline void print_time() {
//    time_t lt = time(NULL);
//    std::cout << ctime(&lt) << std::endl;
//}

//inline char* mystrcat(char *dst, const char *src) {
//    int n = (dst != 0 ? strlen(dst) : 0);
//    dst = (char*)realloc(dst, n + strlen(src) + 1);
//    strcat(dst, src);
//    return dst;
//}

//inline char* mystrdup(const char *src) {
//    char *dst = (char*)malloc(strlen(src) + 1);
//    if (dst != NULL) {
//        strcpy(dst, src);
//    }
//    return dst;
//}

//inline int message_callback(void *instance, const char *format, va_list args) {
//    vfprintf(stdout, format, args);
//    fflush(stdout);
//    return 0;
//}

//inline void Free(n3ldg_plus::dtype** p) {
//    if (*p != NULL)
//        free(*p);
//    *p = NULL;
//}

//inline int mod(int v1, int v2) {
//    if (v1 < 0 || v2 <= 0)
//        return -1;
//    else {
//        return v1 % v2;
//    }
//}

//inline void ones(n3ldg_plus::dtype* p, int length) {
//    for (int idx = 0; idx < length; idx++) {
//        p[idx] = 1.0;
//    }
//}

//inline void zeros(n3ldg_plus::dtype* p, int length) {
//    for (int idx = 0; idx < length; idx++) {
//        p[idx] = 0.0;
//    }
//}

//inline n3ldg_plus::dtype logsumexp(n3ldg_plus::dtype a[], int length) {
//    n3ldg_plus::dtype max = a[0];
//    for (int idx = 1; idx < length; idx++) {
//        if (a[idx] > max)
//            max = a[idx];
//    }

//    n3ldg_plus::dtype sum = 0;
//    for (int idx = 0; idx < length; idx++) {
//        sum += exp(a[idx] - max);
//    }

//    return max + log(sum);
//}

//inline n3ldg_plus::dtype logsumexp(const std::vector<n3ldg_plus::dtype>& a) {
//    int length = a.size();
//    n3ldg_plus::dtype max = a[0];
//    for (int idx = 1; idx < length; idx++) {
//        if (a[idx] > max)
//            max = a[idx];
//    }

//    n3ldg_plus::dtype sum = 0;
//    for (int idx = 0; idx < length; idx++) {
//        sum += exp(a[idx] - max);
//    }

//    return max + log(sum);
//}

//inline bool isPunc(std::string thePostag) {
//    if (thePostag.compare("PU") == 0 || thePostag.compare("``") == 0 || thePostag.compare("''") == 0 || thePostag.compare(",") == 0 || thePostag.compare(".") == 0
//            || thePostag.compare(":") == 0 || thePostag.compare("-LRB-") == 0 || thePostag.compare("-RRB-") == 0 || thePostag.compare("$") == 0
//            || thePostag.compare("#") == 0) {
//        return true;
//    } else {
//        return false;
//    }
//}

//inline bool validlabels(const std::string& curLabel) {
//    if (curLabel[0] == '-' && curLabel[curLabel.length() - 1] == '-') {
//        return false;
//    }

//    return true;
//}

//inline std::string cleanLabel(const std::string& curLabel) {
//    if (curLabel.length() > 2 && curLabel[1] == '-') {
//        if (curLabel[0] == 'B' || curLabel[0] == 'b' || curLabel[0] == 'M' || curLabel[0] == 'm' || curLabel[0] == 'E' || curLabel[0] == 'e' || curLabel[0] == 'S'
//                || curLabel[0] == 's' || curLabel[0] == 'I' || curLabel[0] == 'i') {
//            return curLabel.substr(2);
//        }
//    }

//    return curLabel;
//}

//inline bool is_start_label(const std::string& label) {
//    if (label.length() < 3)
//        return false;
//    return (label[0] == 'b' || label[0] == 'B' || label[0] == 's' || label[0] == 'S') && label[1] == '-';
//}

//inline bool is_continue_label(const std::string& label, const std::string& startlabel,
//        int distance) {
//    if (distance == 0) return true;
//    if (label.length() < 3)
//        return false;
//    if (distance != 0 && is_start_label(label))
//        return false;
//    if ((startlabel[0] == 's' || startlabel[0] == 'S') && startlabel[1] == '-')
//        return false;
//    std::string curcleanlabel = cleanLabel(label);
//    std::string startcleanlabel = cleanLabel(startlabel);
//    if (curcleanlabel.compare(startcleanlabel) != 0)
//        return false;

//    return true;
//}

//inline int cmpIntIntPairByValue(const std::pair<int, int> &x, const std::pair<int, int> &y) {
//    return x.second > y.second;
//}

//inline void sortMapbyValue(const std::unordered_map<int, int> &t_map,
//        std::vector<std::pair<int, int> > &t_vec) {
//    t_vec.clear();

//    for (std::unordered_map<int, int>::const_iterator iter = t_map.begin(); iter != t_map.end();
//            iter++) {
//        t_vec.push_back(std::make_pair(iter->first, iter->second));
//    }
//    std::sort(t_vec.begin(), t_vec.end(), cmpIntIntPairByValue);
//}

//inline void replace_char_by_char(std::string &str, char c1, char c2) {
//    std::string::size_type pos = 0;
//    for (; pos < str.size(); ++pos) {
//        if (str[pos] == c1) {
//            str[pos] = c2;
//        }
//    }
//}

//inline void split_bychars(const std::string& str, std::vector<std::string> & vec,
//        const char *sep = " ") { //assert(vec.empty());
//    vec.clear();
//    std::string::size_type pos1 = 0, pos2 = 0;
//    std::string word;
//    while ((pos2 = str.find_first_of(sep, pos1)) != std::string::npos) {
//        word = str.substr(pos1, pos2 - pos1);
//        pos1 = pos2 + 1;
//        if (!word.empty())
//            vec.push_back(word);
//    }
//    word = str.substr(pos1);
//    if (!word.empty())
//        vec.push_back(word);
//}

//inline void clean_str(std::string &str) {
//    std::string blank = " \t\r\n";
//    std::string::size_type pos1 = str.find_first_not_of(blank);
//    std::string::size_type pos2 = str.find_last_not_of(blank);
//    if (pos1 == std::string::npos) {
//        str = "";
//    } else {
//        str = str.substr(pos1, pos2 - pos1 + 1);
//    }
//}

inline bool my_getline(std::ifstream &inf, std::string &line) {
    if (!getline(inf, line))
        return false;
    int end = line.size() - 1;
    while (end >= 0 && (line[end] == '\r' || line[end] == '\n')) {
        line.erase(end--);
    }

    return true;
}

//inline void str2uint_vec(const std::vector<std::string> &vecStr,
//        std::vector<unsigned int> &vecInt) {
//    vecInt.resize(vecStr.size());
//    int i = 0;
//    for (; i < vecStr.size(); ++i) {
//        vecInt[i] = atoi(vecStr[i].c_str());
//    }
//}

//inline void str2int_vec(const std::vector<std::string> &vecStr, std::vector<int> &vecInt) {
//    vecInt.resize(vecStr.size());
//    int i = 0;
//    for (; i < vecStr.size(); ++i) {
//        vecInt[i] = atoi(vecStr[i].c_str());
//    }
//}

//template<typename A>
//std::string obj2string(const A& a) {
//    std::ostringstream out;
//    out << a;
//    return out.str();
//}

//inline void int2str_vec(const std::vector<int> &vecInt, std::vector<std::string> &vecStr) {
//    vecStr.resize(vecInt.size());
//    int i = 0;
//    for (; i < vecInt.size(); ++i) {
//        std::ostringstream out;
//        out << vecInt[i];
//        vecStr[i] = out.str();
//    }
//}

//inline void join_bystr(const std::vector<std::string> &vec, std::string &str,
//        const std::string &sep) {
//    str = "";
//    if (vec.empty())
//        return;
//    str = vec[0];
//    int i = 1;
//    for (; i < vec.size(); ++i) {
//        str += sep + vec[i];
//    }
//}

//inline void split_bystr(const std::string &str, std::vector<std::string> &vec,
//        const std::string &sep) {
//    vec.clear();
//    std::string::size_type pos1 = 0, pos2 = 0;
//    std::string word;
//    while ((pos2 = str.find(sep, pos1)) != std::string::npos) {
//        word = str.substr(pos1, pos2 - pos1);
//        pos1 = pos2 + sep.size();
//        if (!word.empty())
//            vec.push_back(word);
//    }
//    word = str.substr(pos1);
//    if (!word.empty())
//        vec.push_back(word);
//}

//inline void split_pair_vector(const std::vector<std::pair<int, std::string> > &vecPair,
//        std::vector<int> &vecInt, std::vector<std::string> &vecStr) {
//    int i = 0;
//    vecInt.resize(vecPair.size());
//    vecStr.resize(vecPair.size());
//    for (; i < vecPair.size(); ++i) {
//        vecInt[i] = vecPair[i].first;
//        vecStr[i] = vecPair[i].second;
//    }
//}

inline void split_bychar(const std::string& str, std::vector<std::string>& vec,
        const char separator = ' ') {
    assert(vec.empty());
    vec.clear();
    std::string::size_type pos1 = 0, pos2 = 0;
    std::string word;
    while ((pos2 = str.find_first_of(separator, pos1)) != std::string::npos) {
        word = str.substr(pos1, pos2 - pos1);
        pos1 = pos2 + 1;
        if (!word.empty())
            vec.push_back(word);
    }
    word = str.substr(pos1);
    if (!word.empty())
        vec.push_back(word);
}

//inline void string2pair(const std::string& str, std::pair<std::string, std::string>& pairStr,
//        const char separator = '/') {
//    std::string::size_type pos = str.find_last_of(separator);
//    if (pos == std::string::npos) {
//        std::string tmp = str + "";
//        clean_str(tmp);
//        pairStr.first = tmp;
//        pairStr.second = "";
//    } else {
//        std::string tmp = str.substr(0, pos);
//        clean_str(tmp);
//        pairStr.first = tmp;
//        tmp = str.substr(pos + 1);
//        clean_str(tmp);
//        pairStr.second = tmp;
//    }
//}

//void convert_to_pair(vector<string>& vecString, vector<pair<string, string> >& vecPair) {
//    assert(vecPair.empty());
//    int size = vecString.size();
//    string::size_type cur;
//    string strWord, strPos;
//    for (int i = 0; i < size; ++i) {
//        cur = vecString[i].find('/');

//        if (cur == string::npos) {
//            strWord = vecString[i].substr(0);
//            strPos = "";
//        } else if (cur == vecString[i].size() - 1) {
//            strWord = vecString[i].substr(0, cur);
//            strPos = "";
//        } else {
//            strWord = vecString[i].substr(0, cur);
//            strPos = vecString[i].substr(cur + 1);
//        }

//        vecPair.push_back(pair<string, string>(strWord, strPos));
//    }
//}

//void split_to_pair(const string& str, vector<pair<string, string> >& vecPair) {
//    assert(vecPair.empty());
//    vector<string> vec;
//    split_bychar(str, vec);
//    convert_to_pair(vec, vecPair);
//}

//void chomp(string& str) {
//    string white = " \t\n";
//    string::size_type pos1 = str.find_first_not_of(white);
//    string::size_type pos2 = str.find_last_not_of(white);
//    if (pos1 == string::npos || pos2 == string::npos) {
//        str = "";
//    } else {
//        str = str.substr(pos1, pos2 - pos1 + 1);
//    }
//}

//int common_substr_len(string str1, string str2) {
//    string::size_type minLen;
//    if (str1.length() < str2.length()) {
//        minLen = str1.length();
//    } else {
//        minLen = str2.length();
//        str1.swap(str2); //make str1 the shorter string
//    }

//    string::size_type maxSubstrLen = 0;
//    string::size_type posBeg;
//    string::size_type substrLen;
//    string sub;
//    for (posBeg = 0; posBeg < minLen; posBeg++) {
//        for (substrLen = minLen - posBeg; substrLen > 0; substrLen--) {
//            sub = str1.substr(posBeg, substrLen);
//            if (str2.find(sub) != string::npos) {
//                if (maxSubstrLen < substrLen) {
//                    maxSubstrLen = substrLen;
//                }

//                if (maxSubstrLen >= minLen - posBeg - 1) {
//                    return maxSubstrLen;
//                }
//            }
//        }
//    }
//    return 0;
//}

//int get_char_index(string& str) {
//    assert(str.size() == 2);
//    return ((unsigned char)str[0] - 176) * 94 + (unsigned char)str[1] - 161;
//}

//bool is_chinese_char(string& str) {
//    if (str.size() != 2) {
//        return false;
//    }
//    int index = ((unsigned char)str[0] - 176) * 94 + (unsigned char)str[1] - 161;
//    if (index >= 0 && index < 6768) {
//        return true;
//    } else {
//        return false;
//    }
//}

//int find_GB_char(const string& str, string wideChar, int begPos) {
//    assert(wideChar.size() == 2 && wideChar[0] < 0); //is a GB char
//    int strLen = str.size();

//    if (begPos >= strLen) {
//        return -1;
//    }

//    string GBchar;
//    for (int i = begPos; i < strLen - 1; i++) {
//        if (str[i] < 0) { //is a GB char
//            GBchar = str.substr(i, 2);
//            if (GBchar == wideChar)
//                return i;
//            else
//                i++;
//        }
//    }
//    return -1;
//}

//void split_by_separator(const string& str, vector<string>& vec, const string separator) {
//    assert(vec.empty());
//    string::size_type pos1 = 0, pos2 = 0;
//    string word;

//    while ((pos2 = find_GB_char(str, separator, pos1)) != -1) {
//        word = str.substr(pos1, pos2 - pos1);
//        pos1 = pos2 + separator.size();
//        if (!word.empty())
//            vec.push_back(word);
//    }
//    word = str.substr(pos1);
//    if (!word.empty())
//        vec.push_back(word);
//}

//string word(string& word_pos) {
//    return word_pos.substr(0, word_pos.find("/"));
//}

//bool is_ascii_string(string& word) {
//    for (unsigned int i = 0; i < word.size(); i++) {
//        if (word[i] < 0) {
//            return false;
//        }
//    }
//    return true;
//}

//bool is_startwith(const string& word, const string& prefix) {
//    if (word.size() < prefix.size())
//        return false;
//    for (unsigned int i = 0; i < prefix.size(); i++) {
//        if (word[i] != prefix[i]) {
//            return false;
//        }
//    }
//    return true;
//}


//void remove_beg_end_spaces(string &str) {
//    clean_str(str);
//}

//void split_bystr(const string &str, vector<string> &vec, const char *sep) {
//    split_bystr(str, vec, string(sep));
//}

//string tolowcase(const string& word) {
//    string newword;
//    for (unsigned int i = 0; i < word.size(); i++) {
//        if (word[i] > 'A' && word[i] < 'Z') {
//            char c = word[i] - 'A' + 'a';
//            newword = newword + 'a' + c;
//        } else {
//            newword = newword + word[i];
//        }
//    }
//    return newword;
//}


//struct segIndex {
//    int start;
//    int end;
//    string label;
//};


//void getSegs(const vector<string>& labels, vector<segIndex>& segs) {
//    int idx, idy, endpos;
//    segIndex seg;
//    idx = 0;
//    segs.clear();
//    while (idx < labels.size()) {
//        if (is_start_label(labels[idx])) {
//            idy = idx;
//            endpos = -1;
//            while (idy < labels.size()) {
//                if (!is_continue_label(labels[idy], labels[idx], idy - idx)) {
//                    endpos = idy - 1;
//                    break;
//                }
//                endpos = idy;
//                idy++;
//            }
//            seg.start = idx;
//            seg.end = endpos;
//            seg.label = cleanLabel(labels[idx]);
//            segs.push_back(seg);
//            idx = endpos;
//        }
//        idx++;
//    }
//}

//template<typename A>
//void clearVec(vector<vector<A> >& bivec) {
//    int count = bivec.size();
//    for (int idx = 0; idx < count; idx++) {
//        bivec[idx].clear();
//    }
//    bivec.clear();
//}

//template<typename A>
//void clearVec(vector<vector<vector<A> > >& trivec) {
//    int count1, count2;
//    count1 = trivec.size();
//    for (int idx = 0; idx < count1; idx++) {
//        count2 = trivec[idx].size();
//        for (int idy = 0; idy < count2; idy++) {
//            trivec[idx][idy].clear();
//        }
//        trivec[idx].clear();
//    }
//    trivec.clear();
//}

//template<typename A>
//void resizeVec(vector<vector<A> >& bivec, const int& size1, const int& size2) {
//    bivec.resize(size1);
//    for (int idx = 0; idx < size1; idx++) {
//        bivec[idx].resize(size2);
//    }
//}

//template<typename A>
//void resizeVec(vector<vector<vector<A> > >& trivec, const int& size1, const int& size2, const int& size3) {
//    trivec.resize(size1);
//    for (int idx = 0; idx < size1; idx++) {
//        trivec[idx].resize(size2);
//        for (int idy = 0; idy < size2; idy++) {
//            trivec[idx][idy].resize(size3);
//        }
//    }
//}

//template<typename A>
//void assignVec(vector<A>& univec, const A& a) {
//    int count = univec.size();
//    for (int idx = 0; idx < count; idx++) {
//        univec[idx] = a;
//    }
//}

//template<typename A>
//void assignVec(vector<vector<A> >& bivec, const A& a) {
//    int count1, count2;
//    count1 = bivec.size();
//    for (int idx = 0; idx < bivec.size(); idx++) {
//        count2 = bivec[idx].size();
//        for (int idy = 0; idy < count2; idy++) {
//            bivec[idx][idy] = a;
//        }
//    }
//}

//template<typename A>
//void assignVec(vector<vector<vector<A> > >& trivec, const A& a) {
//    int count1, count2, count3;
//    count1 = trivec.size();
//    for (int idx = 0; idx < count1; idx++) {
//        count2 = trivec[idx].size();
//        for (int idy = 0; idy < count2; idy++) {
//            count3 = trivec[idx][idy].size();
//            for (int idz = 0; idz < count3; idz++) {
//                trivec[idx][idy][idz] = a;
//            }
//        }
//    }
//}


//template<typename A>
//void addAllItems(vector<A>& target, const vector<A>& sources) {
//    int count = sources.size();
//    for (int idx = 0; idx < count; idx++) {
//        target.push_back(sources[idx]);
//    }
//}


//int cmpStringIntPairByValue(const pair<string, int> &x, const pair<string, int> &y) {
//    return x.second > y.second;
//}


template <typename T, typename S>
std::vector<S *> toPointers(std::vector<T> &v, int size) {
    std::vector<S *> pointers;
    for (int i = 0; i < size; ++i) {
        pointers.push_back(&v.at(i));
    }
    return pointers;
}


template <typename T, typename S>
std::vector<S *> toPointers(std::vector<T> &v) {
    return toPointers<T, S>(v, v.size());
}


// for lowercase English only
bool isPunctuation(const std::string &text) {
	for (char c : text) {
		if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')) {
			return false;
		}
	}
	return true;
}

bool isEqual(n3ldg_plus::dtype a, n3ldg_plus::dtype b) {
    float c = a - b;
    if (c < 0.001 && c > -0.001) {
        return true;
    }
    c = c / a;
    return c < 0.001 && c > -0.001;
}

size_t typeSignature(void *p) {
    auto addr = reinterpret_cast<uintptr_t>(p);
#if SIZE_MAX < UINTPTR_MAX
    addr %= SIZE_MAX;
#endif
    return addr;
}

#define n3ldg_assert(assertion, message) \
  if (!(assertion)) {\
    std::cerr << message << endl;\
    abort(); \
  }

template <typename T, typename S>
std::vector<T> transferVector(const std::vector<S> &src_vector,
        const std::function<T(const S&)> &transfer) {
    std::vector<T> result;
    for (const S& src : src_vector) {
        result.push_back(transfer(src));
    }
    return result;
}

}

#endif
