#include <iostream>
#include <fstream>

#include "md5.hpp"
#include "rng.hpp"

int main(int argc, char** argv)
{
	rng target;
	target.Seed();
	
	std::string output_path = argv[1];
	std::ofstream file(output_path, std::ofstream::out);
	int output_size = stoi(argv[2]);
	for (int i = 0; i < output_size; ++i)
	{
		using DataType = uint16_t;
		DataType value = target.get<DataType>();
		
		for (int bit = 0; bit < sizeof(DataType)*8; ++bit)
		{
			file << ((value & (1<<bit)) == 0)?0:1;
		}
		file << std::endl;
		file.flush();
	}
	file.flush();
	file.close();

	return 0;
}
