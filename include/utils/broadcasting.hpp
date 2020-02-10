#ifndef MLKIT_BROADCASTING_HPP
#define MLKIT_BROADCASTING_HPP
namespace mk{
    namespace utils{
        inline af::array badd(const af::array &lhs, const af::array &rhs){
            return lhs + rhs;
        }

        inline af::array bsub(const af::array &lhs, const af::array &rhs){
            return lhs - rhs;
        }

        inline af::array bmul(const af::array &lhs, const af::array &rhs){
            return lhs * rhs;
        }

        inline af::array bdiv(const af::array &lhs, const af::array &rhs){
            return lhs / rhs;
        }

        inline af::array bdis(const af::array &lhs, const af::array &rhs){
            return af::sqrt(lhs * lhs - 2 * lhs * rhs + rhs * rhs);
        }
    }
}
#endif