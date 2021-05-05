#pragma once

#include <memory>
#include <vector>
#include <chrono>

class rng
{
public:
	void Seed(size_t length = 4096, uint64_t seed = 0)
	{
		_length = length;
		_index_loc = 0;
		seed_time = std::chrono::high_resolution_clock::now();

		
		_stream.resize(length);
		for (auto&& value : _stream)
		{
			if (seed < 1000)
			{
				auto now = std::chrono::high_resolution_clock::now();
				seed = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
			}
			
			value = seed;
			seed = (uint64_t)(seed * seed);
		}
	}
	
	template<typename T>
	uint64_t get()
	{
		constexpr int min_step_size = 32;
		constexpr int max_step_size = 64;
		auto now = std::chrono::high_resolution_clock::now();
		auto diff = now - seed_time;
		int shift_count = (min_step_size + diff.count()%(max_step_size-min_step_size));
		
		T output;
		auto* data = reinterpret_cast<uint8_t*>(_stream.data());
		uint8_t md5_out[16];
		md5bin(data + _index_loc, sizeof(uint64_t) * min_step_size, md5_out);
		std::memcpy(&output, md5_out + 16 - sizeof(T),sizeof(T));
		_index_loc += shift_count;
		std::cout << (uint64_t)output << std::endl;
		
		if (_index_loc + max_step_size >= _length)
		{
			Seed(_length);
		}
		
		return output;
	}
	
private:
	std::chrono::high_resolution_clock::time_point seed_time;
	std::vector<uint64_t> _stream;
	int _length;
	int _index_loc;
};