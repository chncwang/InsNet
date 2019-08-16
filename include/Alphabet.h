#ifndef _ALPHABET_
#define _ALPHABET_

#include "MyLib.h"
#include "serializable.h"
#include <boost/format.hpp>
#include <string>

/*
 please check to ensure that m_size not exceeds the upbound of int
 */

/*
  This class serializes feature from string to int.
  Index starts from 0.
*/

/**
 * The basic class of quark class.
 *  @param  std::string        String class name to be used.
 *  @param  int         ID class name to be used.
 *  @author Naoaki Okazaki
 */
class basic_quark : public N3LDGSerializable {
    static const  int max_capacity = 10000000;
public:
    typedef unordered_map<std::string, int> StringToId;
    typedef std::vector<std::string> IdToString;

    StringToId m_string_to_id;
    IdToString m_id_to_string;
    int m_size;

    /**
     * Map a string to its associated ID.
     *  If string-to-integer association does not exist, allocate a new ID.
     *  @param  str         String value.
     *  @return           Associated ID for the string value.
     */
    int operator[](const std::string& str) {
        StringToId::const_iterator it = m_string_to_id.find(str);
        if (it != m_string_to_id.end()) {
            return it->second;
        } else {
            cerr << str << " not found" << endl;
            abort();
        }
    }


    /**
     * Convert ID value into the associated string value.
     *  @param  qid         ID.
     *  @param  def         Default value if the ID was out of range.
     *  @return           String value associated with the ID.
     */
    const std::string& from_id(const int& qid) const {
        if (qid < 0 || m_size <= qid) {
            cerr << "qid:" << qid << endl;
            abort();
        } else {
            return m_id_to_string[qid];
        }
    }

    int insert_string(const std::string& str) {
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

    int from_string(const std::string& str) const {
        StringToId::const_iterator it = m_string_to_id.find(str);
        if (it != m_string_to_id.end()) {
            return it->second;
        } else {
            cerr << str << " not found" << endl;
            abort();
        }
    }

    bool find_string(const string &str) const {
        return m_string_to_id.find(str) != m_string_to_id.end();
    }

    size_t size() const {
        return m_size;
    }

    void read(std::ifstream &inf) {
        string featKey;
        int featId;
        inf >> m_size;
        for (int i = 0; i < m_size; ++i) {
            inf >> featKey >> featId;
            m_string_to_id[featKey] = i;
            m_id_to_string.push_back(featKey);
            assert(featId == i);
        }
    }

    void write(std::ofstream &outf) const {
        outf << m_size << std::endl;
        for (int i = 0; i < m_size; i++) {
            outf << m_id_to_string[i] << " " << i << std::endl;
        }
    }

    void init(const vector<string> &word_list) {
        m_size = word_list.size();
        m_id_to_string = word_list;
        int i = 0;
        for (const string &w : word_list) {
            m_string_to_id.insert(make_pair(w, i++));
        }
    }

    void init(const unordered_map<string, int>& elem_stat, int cutOff = 0) {
        unordered_map<string, int>::const_iterator elem_iter;
        for (elem_iter = elem_stat.begin(); elem_iter != elem_stat.end(); elem_iter++) {
            if (elem_iter->second > cutOff) {
                insert_string(elem_iter->first);
            }
        }
    }

    // initial by a file (first column), always an embedding file
    void init(const string& inFile, bool bUseUnknown = true) {
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

    Json::Value toJson() const override {
        Json::Value json;
        json["m_string_to_id"] = ::toJson(m_string_to_id);
        json["m_id_to_string"] = ::toJson(m_id_to_string);
        json["m_size"] = m_size;
        return json;
    }

    void fromJson(const Json::Value &json) override {
        m_string_to_id = intMapFromJson(json["m_string_to_id"]);
        m_id_to_string = stringVectorFromJson(json["m_id_to_string"]);
        m_size = json["m_size"].asInt();
    }
};

typedef basic_quark Alphabet;

#endif

