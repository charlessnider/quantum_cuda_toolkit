/* UTILITIES FOR WRITING, READING MATRICES FROM TEXT FILES */
//
//      void write_array_to_file_T(T* matrix_to_write, string name_of_output_file, int number_of_elements)
//          write an array of type T (C, Z = complex float, double, S, D = float, double) to a text file
//              matrices should be in memory in COLUMN MAJOR order (i,j) -> k = dim * j + i
//              cuBLAS routines use column major order, ugh
//          text file name = "name_of_output_file.txt" if real valued matrix
//          two outputs for complex matrices: "real_name_of_output_file.txt", "imag_name_of_output_file.txt"
//
//      void read_array_from_file_T(T* pointer_to_matrix, string name_of_input_file)
//          read an array of type T (C, Z = complex float, double, S, D = float, double) from a text file
//              matrices should be in memory in COLUMN MAJOR order (i,j) -> k = dim * j + i
//              cuBLAS routines use column major order, ugh
//          for complex data types, reads from text files "real_name_of_input_file.txt" and "imag_name_of_input_file.txt", for real data types reads from "name_of_input_file.txt"
//
//      void print_complex(cuFloatComplex val)
//          prints a complex value, cuz i'm lazy and don't wanna write it

void print_complex(cuFloatComplex val){
    std::cout << cuCrealf(val) << " + " << cuCimagf(val) << "i\n";
}

// WRITE MATRICES, VECTORS TO FILES

void write_array_to_file_C(cuFloatComplex* M, std::string M_name, int dim){

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
        r_output << cuCrealf(M[i]) << "\n";
        i_output << cuCimagf(M[i]) << "\n";
    }

}

void write_array_to_file_Z(cuDoubleComplex* M, std::string M_name, int dim){

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
        r_output << cuCreal(M[i]) << "\n";
        i_output << cuCimag(M[i]) << "\n";
    }

}

void write_array_to_file_S(float* M, std::string M_name, int dim){

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
        output << M[i] << "\n";
    }

}

void write_array_to_file_D(double* M, std::string M_name, int dim){

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
        output << M[i] << "\n";
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

void read_array_from_file_S(float* M, std::string M_name){

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