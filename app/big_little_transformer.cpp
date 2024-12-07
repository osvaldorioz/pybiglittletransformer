#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <Eigen/Dense>
#include <vector>

namespace py = pybind11;
using Eigen::MatrixXd;
using Eigen::VectorXd;

// Clase para la capa de atención
class AttentionLayer {
public:
    AttentionLayer(int input_dim) {
        weights = MatrixXd::Random(input_dim, input_dim);
        biases = VectorXd::Zero(input_dim);
    }

    MatrixXd forward(const MatrixXd &input) {
        MatrixXd scores = input * weights;
        scores.rowwise() += biases.transpose();
        return scores.array().exp() / scores.array().exp().colwise().sum();
    }

private:
    MatrixXd weights;
    VectorXd biases;
};

// Clase para la rama "Little" (ligera)
class LittleBranch {
public:
    LittleBranch(int input_dim, int output_dim) {
        dense_weights = MatrixXd::Random(input_dim, output_dim);
        dense_biases = VectorXd::Zero(output_dim);
    }

    MatrixXd forward(const MatrixXd &input) {
        return (input * dense_weights).rowwise() + dense_biases.transpose();
    }

private:
    MatrixXd dense_weights;
    VectorXd dense_biases;
};

// Clase para la rama "Big" (compleja)
class BigBranch {
public:
    BigBranch(int input_dim, int output_dim)
        : attention(input_dim) { // Inicializamos AttentionLayer aquí
        dense_weights = MatrixXd::Random(input_dim, output_dim);
        dense_biases = VectorXd::Zero(output_dim);
    }

    MatrixXd forward(const MatrixXd &input) {
        MatrixXd attended = attention.forward(input);
        return (attended * dense_weights).rowwise() + dense_biases.transpose();
    }

private:
    AttentionLayer attention; // Este necesita ser inicializado explícitamente
    MatrixXd dense_weights;
    VectorXd dense_biases;
};

// Modelo Big-Little Transformer
class BigLittleTransformer {
public:
    BigLittleTransformer(int input_dim, int output_dim)
        : little_branch(input_dim, output_dim), big_branch(input_dim, output_dim) {}

    MatrixXd forward(const MatrixXd &input) {
        MatrixXd little_output = little_branch.forward(input);
        MatrixXd big_output = big_branch.forward(input);
        return (little_output + big_output) / 2.0; // Combinación simple de las dos ramas
    }

private:
    LittleBranch little_branch;
    BigBranch big_branch;
};

// Enlace Pybind11
PYBIND11_MODULE(big_little_transformer, m) {
    py::class_<BigLittleTransformer>(m, "BigLittleTransformer")
        .def(py::init<int, int>())
        .def("forward", [](BigLittleTransformer &self, py::array_t<double> input_array) {
            py::buffer_info buf = input_array.request();
            if (buf.ndim != 2) {
                throw std::runtime_error("Input debe ser un array 2D.");
            }

            MatrixXd input = Eigen::Map<MatrixXd>((double *)buf.ptr, buf.shape[0], buf.shape[1]);
            MatrixXd output = self.forward(input);

            return py::array_t<double>(
                {output.rows(), output.cols()},
                {sizeof(double) * output.cols(), sizeof(double)},
                output.data());
        });
}
