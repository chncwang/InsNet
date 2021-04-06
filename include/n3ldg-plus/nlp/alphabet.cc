#include "n3ldg-plus/nlp/alphabet.h"
#include <iostream>
#include "n3ldg-plus/base/def.h"
#include "n3ldg-plus/util/util.h"
#include "fmt/core.h"

using std::ifstream;
using std::string;
using std::vector;
using std::cerr;
using std::endl;
using std::ofstream;
using std::unordered_map;
using std::pair;

namespace n3ldg_plus {

int basic_quark::operator[](const string& str) {
    StringToId::const_iterator it = m_string_to_id.find(str);
    if (it != m_string_to_id.end()) {
        return it->second;
    } else {
        cerr << str << " not found" << endl;
        abort();
    }
}

const string& basic_quark::from_id(const int& qid) const {
    if (qid < 0 || m_size <= qid) {
        cerr << "qid:" << qid << endl;
        abort();
    } else {
        return m_id_to_string[qid];
    }
}

int basic_quark::insert_string(const string& str) {
    StringToId::const_iterator it = m_string_to_id.find(str);
    if (it != m_string_to_id.end()) {
        return it->second;
    } else {
        int newid = m_size;
        m_id_to_string.push_back(str);
        m_string_to_id.insert(pair<string, int>(str, newid));
        m_size++;
        return newid;
    }
}

int basic_quark::from_string(const string& str) const {
    StringToId::const_iterator it = m_string_to_id.find(str);
    if (it != m_string_to_id.end()) {
        return it->second;
    } else if (str == UNKNOWN_WORD) {
        return -1;
    } else {
        cerr << str << " not found" << endl;
        abort();
    }
}

void basic_quark::read(ifstream &inf) {
    string featKey;
    int featId;
    inf >> m_size;
    for (int i = 0; i < m_size; ++i) {
        inf >> featKey >> featId;
        m_string_to_id[featKey] = i;
        m_id_to_string.push_back(featKey);
        if (featId != i) {
            cerr << fmt::format("basic_quark read - featId:{} i:{}\n", featId, i);
            abort();
        }
    }
}

void basic_quark::write(ofstream &outf) const {
    outf << m_size << endl;
    for (int i = 0; i < m_size; i++) {
        outf << m_id_to_string[i] << " " << i << endl;
    }
}

void basic_quark::init(const vector<string> &word_list) {
    m_size = word_list.size();
    m_id_to_string = word_list;
    int i = 0;
    for (const string &w : word_list) {
        m_string_to_id.insert(make_pair(w, i++));
    }
}

void basic_quark::init(const unordered_map<string, int>& elem_stat, int cutOff) {
    unordered_map<string, int>::const_iterator elem_iter;
    for (elem_iter = elem_stat.begin(); elem_iter != elem_stat.end(); elem_iter++) {
        if (elem_iter->second > cutOff) {
            insert_string(elem_iter->first);
        }
    }
}

void basic_quark::init(const string& inFile, bool bUseUnknown) {
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
        from_string(UNKNOWN_WORD);
    }
}

}
