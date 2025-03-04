#include "pgn_parser.hpp"
#include "state.hpp"



chess_move pgn_parser::parse_san(const board_state &state, std::string san)
{
    std::vector<chess_move> legal_moves = state.get_all_legal_moves(state.get_turn());

    if (san.find("0-0") != std::string::npos || san.find("O-O") != std::string::npos) {
        for (size_t i = 0; i < legal_moves.size(); i++) {
            chess_move mov = legal_moves[i];
            if (state.get_square(mov.from).get_type() == KING && mov.to.get_x() == 6) {
                return mov;
            }
        }
        return chess_move::null_move();
    }

    if (san.find("0-0-0") != std::string::npos || san.find("O-O-O") != std::string::npos) {
        for (size_t i = 0; i < legal_moves.size(); i++) {
            chess_move mov = legal_moves[i];
            if (state.get_square(mov.from).get_type() == KING && mov.to.get_x() == 2) {
                return mov;
            }
        }
        return chess_move::null_move();
    }


    bool is_promotion = (san.find('=') != std::string::npos);
    bool is_capture = (san.find('x') != std::string::npos);

    char promo_char = '\0';
    if (is_promotion) {
        promo_char = san[san.find('=')+1];
    }

    std::string stop_chars = "=+# ";
    size_t stop_char_pos = san.length();
    for (size_t i = 0; i < san.length(); i++) {
        if (stop_chars.find(san[i]) != std::string::npos) {
            stop_char_pos = i;
            break;
        }
    }

    if (stop_char_pos < san.length()) {
        san = san.substr(0, stop_char_pos);
    }
    if (is_capture) {
        san.erase(san.find('x'), 1);
    }

    square_index to_sq;
    to_sq.from_uci(san.substr(san.length() - 2, 2));

    san.erase(san.length()-2, 2);


    std::string files = "abcdefgh";
    std::string ranks = "87654321";
    std::string pieces = "NBRQK";
    piece_type_t piece_types[5] = {KNIGHT, BISHOP, ROOK, QUEEN, KING};

    int from_file = -1;
    int from_rank = -1;

    piece_type_t piece_type = PAWN;
    if (san.length() > 0) {
        auto piece_pos = pieces.find(san[0]);
        auto rank_pos = ranks.find(san[0]);
        if (piece_pos != std::string::npos) {
            piece_type = piece_types[piece_pos];
        } else if (rank_pos != std::string::npos) {
            from_file = files.find(san[0]);
        }
        san.erase(0, 1);
    }

    auto file_pos = files.find(san[0]);
    if (san.length() > 0 && file_pos != std::string::npos) {
        from_file = file_pos;
        san.erase(0, 1);
    }

    auto rank_pos = ranks.find(san[0]);
    if (san.length() > 0 && rank_pos != std::string::npos) {
        from_rank = rank_pos;
        san.erase(0, 1);
    }

    piece_type_t promoted_to_type = EMPTY;
    auto promo_pos = pieces.find(promo_char);
    if (is_promotion && piece_type == PAWN && promo_pos != std::string::npos) {
        promoted_to_type = piece_types[promo_pos];
    }

    for (size_t i = 0; i < legal_moves.size(); i++) {
        chess_move mov = legal_moves[i];

        if (mov.to == to_sq && state.get_square(mov.from).get_type() == piece_type) {

            if ((from_file == -1 || from_file == mov.from.get_x()) &&
                (from_rank == -1 || from_rank == mov.from.get_y()) &&
                (!is_promotion || promoted_to_type == mov.promotion)) {

                return mov;
            }
        }
    }

    return chess_move::null_move();
}

std::vector<chess_move> pgn_parser::parse_pgn(std::string pgn_text)
{
    int t0 = 0;
    int t1 = 0;

    int t0_start = 0;
    int t1_start = 0;

    for (size_t i = 0; i < pgn_text.length(); i++) {
        if (pgn_text[i] == '\n') {
            pgn_text[i] = ' ';
        }
        if (pgn_text[i] == '[') {
            if (t0 == 0) { t0_start = i; }
            t0++;
        }
        if (pgn_text[i] == ']') {
            if (t0 == 1) {
                pgn_text.erase(t0_start, i-t0_start+1);
                i = t0_start-1;
            }
            t0--;
        }

        if (pgn_text[i] == '{') {
            if (t1 == 0) { t1_start = i; }
            t1++;
        }
        if (pgn_text[i] == '}') {
            if (t1 == 1) {
                pgn_text.erase(t1_start, i-t1_start+1);
                i = t1_start-1;
            }
            t1--;
        }

    }

    for (size_t i = 0; i < pgn_text.length(); i++) {
        if (pgn_text[i] == '.') {
            pgn_text.insert(i+1, " ");
        }
    }

    auto double_space = pgn_text.find("  ");
    while (double_space != std::string::npos) {
        pgn_text.erase(double_space, 1);
        double_space = pgn_text.find("  ");
    }

    while (pgn_text[0] == ' ') {
        pgn_text.erase(0, 1);
    }

    std::vector<chess_move> moves;
    board_state state;
    state.set_initial_state();

    size_t p = pgn_text.find(' ');
    while (p != std::string::npos) {
        std::string token = pgn_text.substr(0, p);
        pgn_text.erase(0, p + 1);
        p = pgn_text.find(' ');

        if (token.find('.') != std::string::npos || token.length() < 2) {
            continue;
        }

        chess_move mov = parse_san(state, token);

        if (mov.valid()) {
            state.make_move(mov);
            moves.push_back(mov);
        }
    }

    return moves;

}


std::string pgn_parser::read_text_file(std::string path)
{
    std::string lines = "";
    std::ifstream file(path);
    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            lines += line + "\n";
        }
    }
    return lines;
}






















































