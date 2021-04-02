#ifndef _ALPHABET_
#define _ALPHABET_

#include "n3ldg-plus/base/serializable.h"
#include <string>
#include <unordered_map>
#include <vector>

namespace n3ldg_plus {
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
    static const  int max_capacity;
public:
    typedef std::unordered_map<std::string, int> StringToId;
    typedef std::vector<std::string> IdToString;

    StringToId m_string_to_id;
    IdToString m_id_to_string;
    int m_size = 0;

    /**
     * Map a string to its associated ID.
     *  If string-to-integer association does not exist, allocate a new ID.
     *  @param  str         String value.
     *  @return           Associated ID for the string value.
     */
    int operator[](const std::string& str);

    /**
     * Convert ID value into the associated string value.
     *  @param  qid         ID.
     *  @param  def         Default value if the ID was out of range.
     *  @return           String value associated with the ID.
     */
    const std::string& from_id(const int& qid) const;

    int insert_string(const std::string& str);

    int from_string(const std::string& str) const;

    bool find_string(const std::string &str) const {
        return m_string_to_id.find(str) != m_string_to_id.end();
    }

    size_t size() const {
        return m_size;
    }

    void read(std::ifstream &inf);

    void write(std::ofstream &outf) const;

    void init(const std::vector<std::string> &word_list);

    void init(const std::unordered_map<std::string, int>& elem_stat, int cutOff = 0);

    // initial by a file (first column), always an embedding file
    void init(const std::string& inFile, bool bUseUnknown = true);

    template<typename Archive>
    void serialize(Archive &ar) {
        ar(m_string_to_id);
        ar(m_id_to_string);
        ar(m_size);
    }
};

typedef basic_quark Alphabet;
}

#endif
