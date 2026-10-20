// 用 Etterna MinaCalc 导出"逐区间难度"。
//
//   export_itv <map.osu> <out.bin> [rate=1.0] [goal=0.93]
//
// 输出（小端）：
//   uint32 magic = 0x5449434D  ("MCIT")
//   uint32 version = 2
//   uint32 keycount
//   uint32 numitv
//   float  rate, goal
//   float  diffs[2][6][numitv]   手 x {stream, js, hs, jackspeed, cj, tech} x 区间
//   float  pts[2][numitv]        每手每区间点数
//   uint32 n_jack[2]            每手 jack 事件数
//   int32  jack_itv[n_jack]     事件所在区间
//   float  jack_y[n_jack]       事件难度（scaled nps）
//
// 说明：base_adj_diff 与 goal 无关（goal 只进 Chisel），这里仍把 goal 写进头部做溯源。
// 长按只取头部（NoteInfo 没有时长维度），与 Etterna 的口径一致。

#include "MinaCalc/MinaCalc.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace {

bool parse_osu(const std::string& path, int& keycount,
               std::map<long long, unsigned>& rows) {
	std::ifstream in(path);
	if (!in) return false;
	keycount = 4;
	bool in_hit = false;
	std::string line;
	while (std::getline(in, line)) {
		if (!line.empty() && line.back() == '\r') line.pop_back();
		if (line.empty()) continue;
		if (line[0] == '/' && line.size() > 1 && line[1] == '/') continue;
		if (line[0] == '[') {
			in_hit = (line == "[HitObjects]");
			continue;
		}
		if (line.rfind("CircleSize:", 0) == 0) {
			keycount = static_cast<int>(std::strtod(line.c_str() + 11, nullptr));
			if (keycount <= 0) keycount = 4;
			continue;
		}
		if (!in_hit) continue;
		int x = 0, type = 0;
		long long t = 0;
		if (std::sscanf(line.c_str(), "%d,%*d,%lld,%d", &x, &t, &type) != 3) continue;
		int col = static_cast<int>(static_cast<long long>(x) * keycount / 512);
		if (col < 0) col = 0;
		if (col >= keycount) col = keycount - 1;
		rows[t] |= (1u << col);
	}
	return !rows.empty();
}

} // namespace

int main(int argc, char** argv) {
	if (argc < 3) {
		std::fprintf(stderr,
					 "usage: %s <map.osu> <out.bin> [rate=1.0] [goal=0.93]\n",
					 argv[0]);
		return 2;
	}
	const std::string map_path = argv[1];
	const std::string out_path = argv[2];
	const float rate = argc > 3 ? std::strtof(argv[3], nullptr) : 1.0f;
	const float goal = argc > 4 ? std::strtof(argv[4], nullptr) : 0.93f;

	int keycount = 4;
	std::map<long long, unsigned> rows;
	if (!parse_osu(map_path, keycount, rows)) {
		std::fprintf(stderr, "failed to parse %s\n", map_path.c_str());
		return 3;
	}
	if (keycount != 4) {
		std::fprintf(stderr, "only 4K supported (CircleSize=%d)\n", keycount);
		return 3;
	}

	std::vector<NoteInfo> ni;
	ni.reserve(rows.size());
	const long long t0 = rows.begin()->first;
	for (const auto& kv : rows) {
		NoteInfo info{};
		info.notes = kv.second;
		info.rowTime = static_cast<float>((kv.first - t0) / 1000.0);
		ni.push_back(info);
	}

	Calc calc;
	calc.ssr = false; // 谱面侧：不做 40 封顶 / 低准确率降权
	const std::vector<float> skillsets = calc.CalcMain(ni, rate, goal);
	const int n = calc.numitv;
	if (n <= 0) {
		std::fprintf(stderr, "numitv=0 (id=%d, junk file?)\n", skillsets.empty());
		return 4;
	}

	const unsigned dims[6] = { Skill_Stream,	 Skill_Jumpstream,
							   Skill_Handstream, Skill_JackSpeed,
							   Skill_Chordjack,	 Skill_Technical };

	std::vector<float> flat;
	flat.reserve(static_cast<size_t>(2) * 6 * n + 2 * n);
	for (int hand = 0; hand < 2; ++hand) {
		for (int d = 0; d < 6; ++d) {
			const std::vector<float>& v = calc.base_adj_diff.at(hand).at(dims[d]);
			for (int itv = 0; itv < n; ++itv) flat.push_back(v.at(itv));
		}
	}
	for (int hand = 0; hand < 2; ++hand) {
		for (int itv = 0; itv < n; ++itv)
			flat.push_back(static_cast<float>(calc.itv_points.at(hand).at(itv)));
	}

	std::FILE* f = std::fopen(out_path.c_str(), "wb");
	if (!f) {
		std::fprintf(stderr, "cannot open %s\n", out_path.c_str());
		return 5;
	}
	const std::uint32_t header[4] = { 0x5449434Du, 2u,
									  static_cast<std::uint32_t>(keycount),
									  static_cast<std::uint32_t>(n) };
	std::fwrite(header, sizeof(header), 1, f);
	const float meta[2] = { rate, goal };
	std::fwrite(meta, sizeof(meta), 1, f);
	std::fwrite(flat.data(), sizeof(float), flat.size(), f);

	// v2: jack 事件（逐行，保留全部信息；itv 用与计算器一致的分桶）
	std::uint32_t n_jack[2] = { 0u, 0u };
	std::vector<std::int32_t> jitv[2];
	std::vector<float> jy[2];
	for (int hand = 0; hand < 2; ++hand) {
		for (const auto& e : calc.jack_diff.at(hand)) {
			const int itv = static_cast<int>((e.first + 0.0005f) / 0.5f);
			if (itv < 0 || itv >= n) continue;
			jitv[hand].push_back(static_cast<std::int32_t>(itv));
			jy[hand].push_back(e.second);
		}
		n_jack[hand] = static_cast<std::uint32_t>(jitv[hand].size());
	}
	std::fwrite(n_jack, sizeof(n_jack), 1, f);
	for (int hand = 0; hand < 2; ++hand) {
		if (n_jack[hand] == 0) continue;
		std::fwrite(jitv[hand].data(), sizeof(std::int32_t), n_jack[hand], f);
		std::fwrite(jy[hand].data(), sizeof(float), n_jack[hand], f);
	}
	std::fclose(f);
	return 0;
}
