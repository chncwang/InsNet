#include "n3ldg-plus/util/util.h"

using std::string;
using std::ifstream;
using std::vector;

namespace n3ldg_plus {

bool my_getline(ifstream &inf, string &line) {
    if (!getline(inf, line))
        return false;
    int end = line.size() - 1;
    while (end >= 0 && (line[end] == '\r' || line[end] == '\n')) {
        line.erase(end--);
    }

    return true;
}

void split_bychar(const string& str, vector<string>& vec, char separator) {
    vec.clear();
    string::size_type pos1 = 0, pos2 = 0;
    string word;
    while ((pos2 = str.find_first_of(separator, pos1)) != string::npos) {
        word = str.substr(pos1, pos2 - pos1);
        pos1 = pos2 + 1;
        if (!word.empty())
            vec.push_back(word);
    }
    word = str.substr(pos1);
    if (!word.empty())
        vec.push_back(word);
}

bool isEqual(dtype a, dtype b) {
    float c = a - b;
    if (c < 0.001 && c > -0.001) {
        return true;
    }
    c = c / a;
    return c < 0.001 && c > -0.001;
}

}
