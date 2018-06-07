/*
 * COPUtils.h
 *
 *  Created on: Jul 20, 2016
 *      Author: mszhang
 */

#ifndef COPUtil_H_
#define COPUtil_H_
#include "MyLib.h"

const static int maxCapacity = 1 << 30;

inline void hash_combine(size_t& seed, const int& v) {
    seed ^= v + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    //seed = 131 * seed + v;
}

inline size_t hash_func(const int& v1, const int& v2) {
    size_t curIndex = 0;
    hash_combine(curIndex, v1);
    hash_combine(curIndex, v2);
    return curIndex;
}

inline size_t hash_func(const int& v1, const int& v2, const int& v3) {
    size_t curIndex = 0;
    hash_combine(curIndex, v1);
    hash_combine(curIndex, v2);
    hash_combine(curIndex, v3);
    return curIndex;
}

inline size_t hash_func(const int& v1, const int& v2, const int& v3, const int& v4) {
    size_t curIndex = 0;
    hash_combine(curIndex, v1);
    hash_combine(curIndex, v2);
    hash_combine(curIndex, v3);
    hash_combine(curIndex, v4);
    return curIndex;
}

inline size_t hash_func(const int& v1, const int& v2, const int& v3, const int& v4, const int& v5) {
    size_t curIndex = 0;
    hash_combine(curIndex, v1);
    hash_combine(curIndex, v2);
    hash_combine(curIndex, v3);
    hash_combine(curIndex, v4);
    hash_combine(curIndex, v5);
    return curIndex;
}

inline size_t hash_func(const int& v1, const int& v2, const int& v3, const int& v4, const int& v5, const int& v6) {
    size_t curIndex = 0;
    hash_combine(curIndex, v1);
    hash_combine(curIndex, v2);
    hash_combine(curIndex, v3);
    hash_combine(curIndex, v4);
    hash_combine(curIndex, v5);
    hash_combine(curIndex, v6);
    return curIndex;
}

inline size_t hash_func(const int& v1, const int& v2, const int& v3, const int& v4, const int& v5, const int& v6, const int& v7) {
    size_t curIndex = 0;
    hash_combine(curIndex, v1);
    hash_combine(curIndex, v2);
    hash_combine(curIndex, v3);
    hash_combine(curIndex, v4);
    hash_combine(curIndex, v5);
    hash_combine(curIndex, v6);
    hash_combine(curIndex, v7);
    return curIndex;
}

struct C1Feat {
  public:
    int id;
    size_t seed;
  public:
    bool operator == (const C1Feat& a) const {
        return (a.id == id);
    }
    void setId(const int& v) {
        id = v;
        seed = id;
    }
    size_t hash_value() const {
        return seed;
    }
};

struct C2Feat {
  public:
    int id1, id2;
    size_t seed;
  public:
    bool operator == (const C2Feat& a) const {
        return (a.id1 == id1 && a.id2 == id2);
    }
    void setId(const int& v1, const int& v2) {
        id1 = v1;
        id2 = v2;
        seed = hash_func(v1, v2);
    }
    std::size_t hash_value() const {
        return seed;
    }
};

struct C3Feat {
  public:
    int id1, id2, id3;
    size_t seed;
  public:
    bool operator == (const C3Feat& a) const {
        return (a.id1 == id1 && a.id2 == id2 && a.id3 == id3);
    }
    void setId(const int& v1, const int& v2, const int& v3) {
        id1 = v1;
        id2 = v2;
        id3 = v3;
        seed = hash_func(v1, v2, v3);
    }
    std::size_t hash_value() const {
        return seed;
    }
};

struct C4Feat {
  public:
    int id1, id2, id3, id4;
    size_t seed;
  public:
    bool operator == (const C4Feat& a) const {
        return (a.id1 == id1 && a.id2 == id2 && a.id3 == id3 && a.id4 == id4);
    }
    void setId(const int& v1, const int& v2, const int& v3, const int& v4) {
        id1 = v1;
        id2 = v2;
        id3 = v3;
        id4 = v4;
        seed = hash_func(v1, v2, v3, v4);
    }
    std::size_t hash_value() const {
        return seed;
    }
};

struct C5Feat {
  public:
    int id1, id2, id3, id4, id5;
    size_t seed;
  public:
    bool operator == (const C5Feat& a) const {
        return (a.id1 == id1 && a.id2 == id2 && a.id3 == id3 && a.id4 == id4 && a.id5 == id5);
    }
    void setId(const int& v1, const int& v2, const int& v3, const int& v4, const int& v5) {
        id1 = v1;
        id2 = v2;
        id3 = v3;
        id4 = v4;
        id5 = v5;
        seed = hash_func(v1, v2, v3, v4, v5);
    }
    std::size_t hash_value() const {
        return seed;
    }
};

struct C6Feat {
  public:
    int id1, id2, id3, id4, id5, id6;
    size_t seed;
  public:
    bool operator == (const C6Feat& a) const {
        return (a.id1 == id1 && a.id2 == id2 && a.id3 == id3 && a.id4 == id4 && a.id5 == id5 && a.id6 == id6);
    }
    void setId(const int& v1, const int& v2, const int& v3, const int& v4, const int& v5, const int& v6) {
        id1 = v1;
        id2 = v2;
        id3 = v3;
        id4 = v4;
        id5 = v5;
        id6 = v6;
        seed = hash_func(v1, v2, v3, v4, v5, v6);
    }
    std::size_t hash_value() const {
        return seed;
    }
};

struct C7Feat {
  public:
    int id1, id2, id3, id4, id5, id6, id7;
    size_t seed;
  public:
    bool operator == (const C7Feat& a) const {
        return (a.id1 == id1 && a.id2 == id2 && a.id3 == id3 && a.id4 == id4 && a.id5 == id5 && a.id6 == id6 && a.id7 == id7);
    }
    void setId(const int& v1, const int& v2, const int& v3, const int& v4, const int& v5, const int& v6, const int& v7) {
        id1 = v1;
        id2 = v2;
        id3 = v3;
        id4 = v4;
        id5 = v5;
        id6 = v6;
        id7 = v7;
        seed = hash_func(v1, v2, v3, v4, v5, v6, v7);
    }
    std::size_t hash_value() const {
        return seed;
    }
};

struct CFeat {
  public:
    vector<int> ids;
    size_t seed;
    int num;
  public:
    bool valid;

    void clearValue() {
        valid = false;
        seed = -1;
        num = 0;
    }
  public:
    bool operator == (const CFeat& a) const {
        if (a.valid == valid && !valid) {
            return true;
        }
        if (a.valid != valid || a.num != num || a.seed != seed) {
            return false;
        }
        for (int idx = 0; idx < num; idx++) {
            if (a.ids[idx] != ids[idx]) {
                return false;
            }
        }
        return true;
    }

    void setId(const int& v1) {
        if (v1 < 0) {
            valid = false;
            seed = -1;
            num = 0;
            return;
        }
        num = 1;
        if (ids.size() != num)ids.resize(num);
        ids[0] = v1;
        seed = v1;
        valid = true;
    }


    void setId(const int& v1, const int& v2) {
        if (v1 < 0 || v2 < 0) {
            valid = false;
            seed = -1;
            num = 0;
            return;
        }
        num = 2;
        if (ids.size() != num)ids.resize(num);
        ids[0] = v1;
        ids[1] = v2;
        seed = hash_func(v1, v2);
        valid = true;
    }

    void setId(const int& v1, const int& v2, const int& v3) {
        if (v1 < 0 || v2 < 0 || v3 < 0) {
            valid = false;
            seed = -1;
            num = 0;
            return;
        }
        num = 3;
        if (ids.size() != num)ids.resize(num);
        ids[0] = v1;
        ids[1] = v2;
        ids[2] = v3;
        seed = hash_func(v1, v2, v3);
        valid = true;
    }

    void setId(const int& v1, const int& v2, const int& v3, const int& v4) {
        if (v1 < 0 || v2 < 0 || v3 < 0 || v4 < 0) {
            valid = false;
            seed = -1;
            num = 0;
            return;
        }
        num = 4;
        if (ids.size() != num)ids.resize(num);
        ids[0] = v1;
        ids[1] = v2;
        ids[2] = v3;
        ids[3] = v4;
        seed = hash_func(v1, v2, v3, v4);
        valid = true;
    }

    void setId(const int& v1, const int& v2, const int& v3, const int& v4, const int& v5) {
        if (v1 < 0 || v2 < 0 || v3 < 0 || v4 < 0 || v5 < 0) {
            valid = false;
            seed = -1;
            num = 0;
            return;
        }
        num = 5;
        if (ids.size() != num)ids.resize(num);
        ids[0] = v1;
        ids[1] = v2;
        ids[2] = v3;
        ids[3] = v4;
        ids[4] = v5;
        seed = hash_func(v1, v2, v3, v4, v5);
        valid = true;
    }

    void setId(const int& v1, const int& v2, const int& v3, const int& v4, const int& v5, const int& v6) {
        if (v1 < 0 || v2 < 0 || v3 < 0 || v4 < 0 || v5 < 0 || v6 < 0) {
            valid = false;
            seed = -1;
            num = 0;
            return;
        }
        num = 6;
        if (ids.size() != num)ids.resize(num);
        ids[0] = v1;
        ids[1] = v2;
        ids[2] = v3;
        ids[3] = v4;
        ids[4] = v5;
        ids[5] = v6;
        seed = hash_func(v1, v2, v3, v4, v5, v6);
        valid = true;
    }
    //at most 7
    void setId(const int& v1, const int& v2, const int& v3, const int& v4, const int& v5, const int& v6, const int& v7) {
        if (v1 < 0 || v2 < 0 || v3 < 0 || v4 < 0 || v5 < 0 || v6 < 0 || v7 < 0) {
            valid = false;
            seed = -1;
            num = 0;
            return;
        }
        num = 7;
        if (ids.size() != num)ids.resize(num);
        ids[0] = v1;
        ids[1] = v2;
        ids[2] = v3;
        ids[3] = v4;
        ids[4] = v5;
        ids[5] = v6;
        ids[6] = v7;
        seed = hash_func(v1, v2, v3, v4, v5, v6, v7);
        valid = true;
    }
    std::size_t hash_value() const {
        return seed;
    }
};


namespace std {
template<>
struct hash < C1Feat > {
  public:
    size_t operator()(const C1Feat& s)const {
        return s.hash_value();
    }
};

template<>
struct hash < C2Feat > {
  public:
    size_t operator()(const C2Feat& s)const {
        return s.hash_value();
    }
};

template<>
struct hash < C3Feat > {
  public:
    size_t operator()(const C3Feat& s)const {
        return s.hash_value();
    }
};


template<>
struct hash < C4Feat > {
  public:
    size_t operator()(const C4Feat& s)const {
        return s.hash_value();
    }
};

template<>
struct hash < C5Feat > {
  public:
    size_t operator()(const C5Feat& s)const {
        return s.hash_value();
    }
};

template<>
struct hash < C6Feat > {
  public:
    size_t operator()(const C6Feat& s)const {
        return s.hash_value();
    }
};


template<>
struct hash < C7Feat > {
  public:
    size_t operator()(const C7Feat& s)const {
        return s.hash_value();
    }
};

template<>
struct hash < CFeat > {
  public:
    size_t operator()(const CFeat& s)const {
        return s.hash_value();
    }
};

};



#endif /* COPUtil_H_ */
