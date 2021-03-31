#include "n3ldg-plus/nlp/alphabet.h"

using std::ifstream;
using std::string;
using std::vector;

namespace n3ldg_plus {

int basic_quark::operator[](const std::string& str) {
    StringToId::const_iterator it = m_string_to_id.find(str);
    if (it != m_string_to_id.end()) {
        return it->second;
    } else {
        std::cerr << str << " not found" << std::endl;
        abort();
    }
}

const std::string& basic_quark::from_id(const int& qid) const {
    if (qid < 0 || m_size <= qid) {
        std::cerr << "qid:" << qid << std::endl;
        abort();
    } else {
        return m_id_to_string[qid];
    }
}

int basic_quark::insert_string(const std::string& str) {
    StringToId::const_iterator it = m_string_to_id.find(str);
    if (it != m_string_to_id.end()) {
        return it->second;
    } else {
        int newid = m_size;
        m_id_to_string.push_back(str);
        m_string_to_id.insert(std::pair<std::string, int>(str, newid));
        m_size++;
        return newid;
    }
}

int basic_quark::from_string(const std::string& str) const {
    StringToId::const_iterator it = m_string_to_id.find(str);
    if (it != m_string_to_id.end()) {
        return it->second;
    } else if (str == unknownkey) {
        return -1;
    } else {
        std::cerr << str << " not found" << std::endl;
        abort();
    }
}

void basic_quark::read(std::ifstream &inf) {
    std::string featKey;
    int featId;
    inf >> m_size;
    for (int i = 0; i < m_size; ++i) {
        inf >> featKey >> featId;
        m_string_to_id[featKey] = i;
        m_id_to_string.push_back(featKey);
        assert(featId == i);
    }
}

void basic_quark::write(std::ofstream &outf) const {
    outf << m_size << std::endl;
    for (int i = 0; i < m_size; i++) {
        outf << m_id_to_string[i] << " " << i << std::endl;
    }
}

void basic_quark::init(const std::vector<std::string> &word_list) {
    m_size = word_list.size();
    m_id_to_string = word_list;
    int i = 0;
    for (const std::string &w : word_list) {
        m_string_to_id.insert(make_pair(w, i++));
    }
}

void basic_quark::init(const std::unordered_map<std::string, int>& elem_stat, int cutOff) {
    std::unordered_map<std::string, int>::const_iterator elem_iter;
    for (elem_iter = elem_stat.begin(); elem_iter != elem_stat.end(); elem_iter++) {
        if (elem_iter->second > cutOff) {
            insert_string(elem_iter->first);
        }
    }
}

void basic_quark::init(const std::string& inFile, bool bUseUnknown) {
    ifstream inf;
    inf.open(inFile.c_str());

    string strLine;
    vector<string> vecInfo;
    while (1) {
        if (!my_getline(inf, strLine)) {
            break;
        }
        if (!strLine.empty()) {
            split_bychar(strLine, vecInfo, ' ');
            from_string(vecInfo[0]);
        }
    }
    if (bUseUnknown) {
        from_string(unknownkey);
    }
}

}
