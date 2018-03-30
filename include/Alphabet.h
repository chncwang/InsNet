#ifndef _ALPHABET_
#define _ALPHABET_

#include "MyLib.h"

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
class basic_quark {
    static const  int max_capacity = 10000000;
  public:
    typedef unordered_map<std::string, int> StringToId;
    typedef std::vector<std::string> IdToString;

    StringToId m_string_to_id;
    IdToString m_id_to_string;
    bool m_b_fixed;
    int m_size;

  public:
    /**
     * Construct.
     */
    basic_quark() {
        clear();

    }

    /**
     * Destruct.
     */
    virtual ~basic_quark() {
    }

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
        } else if (!m_b_fixed) {
            int newid = m_size;
            m_id_to_string.push_back(str);
            m_string_to_id.insert(std::pair<std::string, int>(str, newid));
            m_size++;
            if (m_size >= max_capacity)m_b_fixed = true;
            return newid;
        } else {
            return -1;
        }
    }


    /**
     * Convert ID value into the associated string value.
     *  @param  qid         ID.
     *  @param  def         Default value if the ID was out of range.
     *  @return           String value associated with the ID.
     */
    const std::string& from_id(const int& qid, const std::string& def = "") const {
        if (qid < 0 || m_size <= qid) {
            return def;
        } else {
            return m_id_to_string[qid];
        }
    }



    /**
     * Convert string value into the associated ID value.
     *  @param  str         String value.
     *  @return           ID if any, otherwise -1.
     */
    int from_string(const std::string& str) {
        StringToId::const_iterator it = m_string_to_id.find(str);
        if (it != m_string_to_id.end()) {
            return it->second;
        } else if (!m_b_fixed) {
            int newid = m_size;
            m_id_to_string.push_back(str);
            m_string_to_id.insert(std::pair<std::string, int>(str, newid));
            m_size++;
            if (m_size >= max_capacity)m_b_fixed = true;
            return newid;
        } else {
            return -1;
        }
    }

    void clear() {
        m_string_to_id.clear();
        m_id_to_string.clear();
        m_b_fixed = false;
        m_size = 0;
    }

    void set_fixed_flag(bool bfixed) {
        m_b_fixed = bfixed;
        if (!m_b_fixed && m_size >= max_capacity) {
            m_b_fixed = true;
        }
    }

    bool is_fixed() const {
        return m_b_fixed;
    }

    /**
     * Get the number of string-to-id associations.
     *  @return           The number of association.
     */
    size_t size() const {
        return m_size;
    }


    void read(std::ifstream &inf) {
        clear();
        string featKey;
        int featId;
        inf >> m_size;
        for (int i = 0; i < m_size; ++i) {
            inf >> featKey >> featId;
            m_string_to_id[featKey] = i;
            m_id_to_string.push_back(featKey);
            assert(featId == i);
        }
        if (m_size > 0) {
            set_fixed_flag(true);
        }
    }

    void write(std::ofstream &outf) const {
        outf << m_size << std::endl;
        for (int i = 0; i < m_size; i++) {
            outf << m_id_to_string[i] << " " << i << std::endl;
        }
    }

    void initial(const unordered_map<string, int>& elem_stat, int cutOff = 0) {
        clear();
        unordered_map<string, int>::const_iterator elem_iter;
        for (elem_iter = elem_stat.begin(); elem_iter != elem_stat.end(); elem_iter++) {
            if (elem_iter->second > cutOff) {
                from_string(elem_iter->first);
            }
        }
        set_fixed_flag(true);
    }

    // initial by a file (first column), always an embedding file
    void initial(const string& inFile, bool bUseUnknown = true) {
        clear();
        ifstream inf;
        if (inf.is_open()) {
            inf.close();
            inf.clear();
        }
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
        if (m_size > 0) {
            set_fixed_flag(true);
        }
    }

};

typedef basic_quark Alphabet;
typedef basic_quark*  PAlphabet;

#endif

