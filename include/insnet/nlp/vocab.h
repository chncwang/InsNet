#ifndef INSNET_VOCAB_H
#define INSNET_VOCAB_H

#include <string>
#include <unordered_map>
#include <vector>

namespace insnet {

class Vocab {
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

    [[deprecated]]
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

}

#endif
