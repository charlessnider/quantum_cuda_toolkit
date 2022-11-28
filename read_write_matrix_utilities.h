/* UTILITIES FOR WRITING, READING MATRICES FROM TEXT FILES */
//
//      void write_matrix_to_file_T(T* matrix_to_write, string name_of_output_file, int size_of_matrix)
//          write a matrix of type T (C, Z = complex float, double, F, D = float, double) to a text file
//              matrix should be in memory in ROW MAJOR order (i,j) -> k = dim * i + j
//          text file name = "name_of_output_file.txt" if real valued matrix
//          two outputs for complex matrices: "real_name_of_output_file.txt", "imag_name_of_output_file.txt"
//
//      void write_vector_to_file_T(T* vector_to_write, string name_of_output_file, int size_of_vector)
//          same as above, but for a vector (1d) rather than matrix (2d)
//
//      void read_array_from_file_T(T* pointer_to_matrix, string name_of_input_file)
//          read a matrix/vector of type T (C, Z = complex float, double, F, D = float, double) from a text file
//              matrix/vector is saved in memory at pointer_to_matrix in ROW MAJOR order (i,j) -> k = dim * i + j
//          for complex data types, reads from text files "real_name_of_input_file.txt" and "imag_name_of_input_file.txt", for real data types reads from "name_of_input_file.txt"
//
//      void print_complex(cuFloatComplex val)
//          prints a complex value, cuz i'm lazy and don't wanna write it

void print_complex(cuFloatComplex val){
    std::cout << cuCrealf(val) << " + " << cuCimagf(val) << "i\n";
}

// WRITE MATRICES, VECTORS TO FILES

void write_matrix_to_file_C(cuFloatComplex* M, std::string M_name, int dim){

    // file extension and prefixes
    std::string txt = ".txt";
    std::string re = "real_";
    std::string im = "imag_";

    // make the real and imaginary file names
    std::string re_name = re + M_name + txt;
    std::string im_name = im + M_name + txt;

    // convert to type for ofstream
    const char* r_output_name = re_name.c_str();
    const char* i_output_name = im_name.c_str();

    // open the files
    std::ofstream r_output;         std::ofstream i_output;
    r_output.open(r_output_name);   i_output.open(i_output_name);

    // write to the files
    for (int i = 0; i < dim; i++)
    {
        for (int j = 0; j < dim; j++)
        {
            r_output << cuCrealf(M[dim * i + j]) << "\n";
            i_output << cuCimagf(M[dim * i + j]) << "\n";
        }
    }

}

void write_matrix_to_file_D(cuDoubleComplex* M, std::string M_name, int dim){

    // file extension and prefixes
    std::string txt = ".txt";
    std::string re = "real_";
    std::string im = "imag_";

    // make the real and imaginary file names
    std::string re_name = re + M_name + txt;
    std::string im_name = im + M_name + txt;

    // convert to type for ofstream
    const char* r_output_name = re_name.c_str();
    const char* i_output_name = im_name.c_str();

    // open the files
    std::ofstream r_output;         std::ofstream i_output;
    r_output.open(r_output_name);   i_output.open(i_output_name);

    // write to the files
    for (int i = 0; i < dim; i++)
    {
        for (int j = 0; j < dim; j++)
        {
            r_output << cuCreal(M[dim * i + j]) << "\n";
            i_output << cuCimag(M[dim * i + j]) << "\n";
        }
    }

}

void write_matrix_to_file_F(float* M, std::string M_name, int dim){

    // file extension and prefixes
    std::string txt = ".txt";

    // make the real and imaginary file names
    std::string name = M_name + txt;

    // convert to type for ofstream
    const char* output_name = name.c_str();

    // open the files
    std::ofstream output;
    output.open(output_name);

    // write to the files
    for (int i = 0; i < dim; i++)
    {
        for (int j = 0; j < dim; j++)
        {
            output << M[dim * i + j] << "\n";
        }
    }

}

void write_matrix_to_file_D(double* M, std::string M_name, int dim){

    // file extension and prefixes
    std::string txt = ".txt";

    // make the real and imaginary file names
    std::string name = M_name + txt;

    // convert to type for ofstream
    const char* output_name = name.c_str();

    // open the files
    std::ofstream output;
    output.open(output_name);

    // write to the files
    for (int i = 0; i < dim; i++)
    {
        for (int j = 0; j < dim; j++)
        {
            output << M[dim * i + j] << "\n";
        }
    }

}

void write_vector_to_file_C(cuFloatComplex* V, std::string V_name, int dim){

    // file extension and prefixes
    std::string txt = ".txt";
    std::string re = "real_";
    std::string im = "imag_";

    // make the real and imaginary file names
    std::string re_name = re + V_name + txt;
    std::string im_name = im + V_name + txt;

    // convert to type for ofstream
    const char* r_output_name = re_name.c_str();
    const char* i_output_name = im_name.c_str();

    // open the files
    std::ofstream r_output;         std::ofstream i_output;
    r_output.open(r_output_name);   i_output.open(i_output_name);

    // write to the files
    for (int i = 0; i < dim; i++)
    {
        r_output << cuCrealf(V[i]) << "\n";
        i_output << cuCimagf(V[i]) << "\n";
    }

}

void write_vector_to_file_Z(cuDoubleComplex * V, std::string V_name, int dim){

    // file extension and prefixes
    std::string txt = ".txt";
    std::string re = "real_";
    std::string im = "imag_";

    // make the real and imaginary file names
    std::string re_name = re + V_name + txt;
    std::string im_name = im + V_name + txt;

    // convert to type for ofstream
    const char* r_output_name = re_name.c_str();
    const char* i_output_name = im_name.c_str();

    // open the files
    std::ofstream r_output;         std::ofstream i_output;
    r_output.open(r_output_name);   i_output.open(i_output_name);

    // write to the files
    for (int i = 0; i < dim; i++)
    {
        r_output << cuCreal(V[i]) << "\n";
        i_output << cuCimag(V[i]) << "\n";
    }

}

void write_vector_to_file_D(double* V, std::string V_name, int dim){

    // file extension and prefixes
    std::string txt = ".txt";

    // make the real and imaginary file names
    std::string name = V_name + txt;

    // convert to type for ofstream
    const char* output_name = name.c_str();

    // open the files
    std::ofstream output;
    output.open(output_name);

    // write to the files
    for (int i = 0; i < dim; i++)
    {
        output << V[i] << "\n";
    }

}

void write_vector_to_file_F(float* V, std::string V_name, int dim){

    // file extension and prefixes
    std::string txt = ".txt";

    // make the real and imaginary file names
    std::string name = V_name + txt;

    // convert to type for ofstream
    const char* output_name = name.c_str();

    // open the files
    std::ofstream output;
    output.open(output_name);

    // write to the files
    for (int i = 0; i < dim; i++)
    {
        output << V[i] << "\n";
    }

}

// READ MATRICES, VECTORES FROM FILE

void read_array_from_file_C(cuFloatComplex* M, std::string M_name){

    // file extension and prefixes
    std::string txt = ".txt";
    std::string re = "real_";
    std::string im = "imag_";

    // make the real and imaginary file names
    std::string re_name = re + M_name + txt;
    std::string im_name = im + M_name + txt;

    // convert to type for ifstream
    const char* r_input_name = re_name.c_str();
    const char* i_input_name = im_name.c_str();

    // open the files
    std::ifstream r_input;         std::ifstream i_input;
    r_input.open(r_input_name);    i_input.open(i_input_name);

    // read from files
    float mij; int idx = 0;
    while (r_input >> mij){

        M[idx] = make_cuFloatComplex(mij, float(0));
        idx += 1;
    }

    // load imaginary part
    idx = 0;
    while (i_input >> mij){

        M[idx] = cuCaddf(M[idx], make_cuFloatComplex(float(0), mij));
        idx += 1;
    }
}

void read_array_from_file_Z(cuDoubleComplex* M, std::string M_name){

    // file extension and prefixes
    std::string txt = ".txt";
    std::string re = "real_";
    std::string im = "imag_";

    // make the real and imaginary file names
    std::string re_name = re + M_name + txt;
    std::string im_name = im + M_name + txt;

    // convert to type for ifstream
    const char* r_input_name = re_name.c_str();
    const char* i_input_name = im_name.c_str();

    // open the files
    std::ifstream r_input;         std::ifstream i_input;
    r_input.open(r_input_name);    i_input.open(i_input_name);

    // read from files
    double mij; int idx = 0;
    while (r_input >> mij){

        M[idx] = make_cuDoubleComplex(mij, double(0));
        idx += 1;
    }

    // load imaginary part
    idx = 0;
    while (i_input >> mij){

        M[idx] = cuCadd(M[idx], make_cuDoubleComplex(double(0), mij));
        idx += 1;
    }
}

void read_array_from_file_F(float* M, std::string M_name){

    // file extension and prefixes
    std::string txt = ".txt";

    // make the real and imaginary file names
    std::string name = M_name + txt;

    // convert to type for ifstream
    const char* input_name = name.c_str();

    // open the files
    std::ifstream input;
    input.open(input_name);

    // read from files
    float mij; int idx = 0;
    while (input >> mij){

        M[idx] = mij;
        idx += 1;
    }
}

void read_array_from_file_D(double* M, std::string M_name){

    // file extension and prefixes
    std::string txt = ".txt";

    // make the real and imaginary file names
    std::string name = M_name + txt;

    // convert to type for ifstream
    const char* input_name = name.c_str();

    // open the files
    std::ifstream input;
    input.open(input_name);

    // read from files
    double mij; int idx = 0;
    while (input >> mij){

        M[idx] = mij;
        idx += 1;
    }
}