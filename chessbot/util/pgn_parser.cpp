#include <sstream>

#include "pgn_parser.hpp"
#include "../state.hpp"
#include "../movegen.hpp"

size_t find_end_of_parentheses(size_t pos, char start_symbol, char end_symbol, std::string &str)
{
    int c = (str[pos] == start_symbol);
    while (c > 0 && pos+1 < str.length()) {
        pos++;
        if (str[pos] == start_symbol) {
            c++;
        } else if (str[pos] == end_symbol) {
            c--;
        }
    }
    return pos;
}

size_t match_str(size_t pos, std::string &str, const std::string &token)
{
    size_t matching_chars = 0;
    for (size_t i = 0; i < token.length() && pos + i < str.length(); i++) {
        if (token[i] == str[pos + i]) {
            matching_chars++;
        } else {
            break;
        }
    }
    return (matching_chars == token.length() ? matching_chars : 0);
}


chess_move pgn_parser::parse_san(const board_state &state, std::string san)
{
    san.erase(std::remove_if(san.begin(), san.end(), [](unsigned char c) { return !std::isprint(c); }), san.end());

    std::vector<chess_move> legal_moves = state.get_all_legal_moves(state.get_turn());

    if (san.find("0-0-0") != std::string::npos || san.find("O-O-O") != std::string::npos) {
        for (size_t i = 0; i < legal_moves.size(); i++) {
            chess_move mov = legal_moves[i];
            if (state.get_square(mov.from).get_type() == KING && mov.to.get_x() == 2) {
                return mov;
            }
        }
        return chess_move::null_move();
    }

    if (san.find("0-0") != std::string::npos || san.find("O-O") != std::string::npos) {
        for (size_t i = 0; i < legal_moves.size(); i++) {
            chess_move mov = legal_moves[i];
            if (state.get_square(mov.from).get_type() == KING && mov.to.get_x() == 6) {
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
            san.erase(0, 1);
        } else if (rank_pos != std::string::npos) {
            from_file = files.find(san[0]);
            san.erase(0, 1);
        }
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

std::vector<chess_move> pgn_parser::parse_moves(std::string pgn_text, std::string startpos)
{
    for (size_t i = 0; i < pgn_text.length(); i++) {
        char end_symbol = '\0';
        if (pgn_text[i] == '[') end_symbol = ']';
        if (pgn_text[i] == '(') end_symbol = ')';
        if (pgn_text[i] == '{') end_symbol = '}';
        if (pgn_text[i] == ';') end_symbol = '\n';
        if (end_symbol != '\0') {
            size_t pend = find_end_of_parentheses(i, pgn_text[i], end_symbol, pgn_text);
            if (end_symbol == '\n') {
                pgn_text.erase(i, pend - i);
            } else {
                pgn_text.erase(i, pend - i + 1);
            }
            i--;
        }
    }

    for (size_t i = 0; i < pgn_text.length(); i++) {
        if (pgn_text[i] == '\n') {
            pgn_text[i] = ' ';
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
    if (startpos != "") {
        state.load_fen(startpos);
    } else {
        state.set_initial_state();
    }

    size_t p = pgn_text.find(' ');
    while (p != std::string::npos) {
        std::string token = pgn_text.substr(0, p);
        pgn_text.erase(0, p + 1);
        p = pgn_text.find(' ');

        if (token.find('.') != std::string::npos || token.length() < 2) {
            continue;
        }

        if (token == "1-0" || token == "0-1" || token == "1/2-1/2" || token == "*") {
            break;
        }

        chess_move mov = parse_san(state, token);

        if (mov.valid()) {
            state.make_move(mov);
            moves.push_back(mov);
        } else {
            std::cout << "Illegal move!" << std::endl;
            std::cout << "Position FEN: " << state.generate_fen() << std::endl;
            std::cout << "Token: " << token << std::endl;
            break;
        }
    }

    return moves;
}



std::vector<std::string> pgn_parser::parse_games(std::string pgn_text)
{
    std::vector<std::string> games;

    std::string end_tokens[4] = {"*", "1-0", "0-1", "1/2-1/2"};

    size_t game_start = 0;
    for (size_t i = 0; i < pgn_text.length(); i++) {
        char end_symbol = '\0';
        if (pgn_text[i] == '[') end_symbol = ']';
        if (pgn_text[i] == '(') end_symbol = ')';
        if (pgn_text[i] == '{') end_symbol = '}';
        if (pgn_text[i] == ';') end_symbol = '\n';
        if (end_symbol != '\0') {
            i = find_end_of_parentheses(i, pgn_text[i], end_symbol, pgn_text);
        }
        size_t end_token_match = 0;
        for (int j = 0; j < 4 && end_token_match == 0; j++) {
            end_token_match = match_str(i, pgn_text, end_tokens[j]);
        }
        if (end_token_match > 0) {
            i += end_token_match+1;
            games.push_back(pgn_text.substr(game_start, i - game_start));
            game_start = i;
        }
    }

    return games;
}


std::vector<pgn_tag> pgn_parser::parse_tags(std::string pgn_text)
{
    std::vector<pgn_tag> tags;

    for (size_t i = 0; i < pgn_text.length(); i++) {
        if (pgn_text[i] != '[') {
            continue;
        }
        size_t end_of_parenthesis = find_end_of_parentheses(i, '[', ']', pgn_text);

        std::string tag_str = pgn_text.substr(i+1, end_of_parenthesis - i - 1);

        std::string tag_name = "";
        std::string tag_value = "";

        size_t space = tag_str.find(' ');
        if (space != std::string::npos) {
            tag_name = tag_str.substr(0, space);
        }


        size_t value_start = tag_str.find('"');
        if (value_start != std::string::npos) {
            size_t value_end = tag_str.find('"', value_start+1);
            if (value_end != std::string::npos) {
                tag_value = tag_str.substr(value_start+1, value_end - value_start - 1);
            }
        }

        tags.push_back(std::make_pair(tag_name, tag_value));

        i = end_of_parenthesis;
    }


    return tags;
}


std::string pgn_parser::get_tag_value(const std::vector<pgn_tag> &tags, std::string tag_name)
{
    for (size_t i = 0; i < tags.size(); i++) {
        if (tags[i].first == tag_name) {
            return tags[i].second;
        }
    }
    return "";
}

void pgn_parser::set_tag_value(std::vector<pgn_tag> &tags, std::string tag_name, std::string tag_value)
{
    for (size_t i = 0; i < tags.size(); i++) {
        if (tags[i].first == tag_name) {
            tags[i].second = tag_value;
            return;
        }
    }
    tags.emplace_back(tag_name, tag_value);
}



std::string pgn_parser::generate_san(board_state &state, chess_move mov)
{
    std::stringstream ss;

    if (mov.is_castling()) {
        if (mov.to.get_x() == 6) {
            ss << "O-O";
        } else {
            ss << "O-O-O";
        }
    } else {

        piece_type_t piece = mov.get_moving_piece().get_type();

        static const char *files = "abcdefgh";
        static const char *ranks = "87654321";
        static const char *pieces = "NBRQK";

        if (piece >= KNIGHT) {
            ss << pieces[piece-KNIGHT];
        }

        bool same_target = false;
        bool same_file = false;
        bool same_rank = false;

        std::vector<chess_move> legal_moves = state.get_all_legal_moves(state.get_turn());

        for (size_t i = 0; i < legal_moves.size(); i++) {
            if (legal_moves[i].from == mov.from) {
                continue;
            }
            if (legal_moves[i].to == mov.to && piece == legal_moves[i].get_moving_piece().get_type()) {
                same_file |= legal_moves[i].from.get_x() == mov.from.get_x();
                same_rank |= legal_moves[i].from.get_y() == mov.from.get_y();

                same_target = true;
            }
        }

        bool write_file = false;
        bool write_rank = false;

        if (same_target && !same_file) {
            write_file = true;
        } else if (same_target && !same_rank) {
            write_rank = true;
        } else if (same_file && same_rank) {
            write_file = true;
            write_rank = true;
        }

        if (piece == PAWN && mov.is_capture()) {
            write_file = true;
        }

        if (write_file) {
            ss << files[mov.from.get_x()];
        }
        if (write_rank) {
            ss << ranks[mov.from.get_y()];
        }
        if (mov.is_capture()) {
            ss << "x";
        }

        ss << mov.to.to_uci();

        if (mov.is_promotion()) {
            ss << "=";
            ss << pieces[mov.promotion-KNIGHT];
        }
    }

    if (state.causes_check(mov, next_turn(state.get_turn()))) {
        unmake_restore restore = state.make_move(mov);

        bool is_checkmate = (state.count_legal_moves(state.get_turn()) == 0);

        state.unmake_move(mov, restore);

        ss << (is_checkmate ? '#' : '+');
    }

    return ss.str();
}


std::string pgn_parser::generate_pgn(const std::vector<pgn_tag> &tags, const std::vector<chess_move> &moves, std::string startpos)
{

    std::string result = "*";

    if (get_tag_value(tags, "Result") != "") {
        result = get_tag_value(tags, "Result");
    }

    std::stringstream pgn_ss;
    std::stringstream moves_ss;

    board_state state;

    if (startpos.length() > 0) {
        state.load_fen(startpos);
    } else {
        state.set_initial_state();
    }

    for (size_t i = 0; i < moves.size(); i++) {
        chess_move mov = moves[i];

        mov.encoded_pieces = move_generator::encode_move_pieces(state, mov);

        if (i % 2 == 0) {
            moves_ss << i/2 + 1 << ".";
        }

        std::string san = generate_san(state, mov);

        moves_ss << san;
        if (i > 0 && (i % 16) == 0) {
            moves_ss << "\n";
        } else {
            moves_ss << " ";
        }

        state.make_move(mov);

        if (san[san.length()-1] == '#') {
            if (state.get_turn() == BLACK) {
                result = "1-0"; //Black gets mated. White wins
            } else {
                result = "0-1"; //White gets mated. Black wins
            }
        } else {
            if (state.count_legal_moves(state.get_turn()) == 0) {
                result = "1/2-1/2"; //Stalemate. Automatic draw
            }
            //Other draw types must be claimed (50 move rule and three fold repetition)
        }
    }

    std::vector<pgn_tag> write_tags = tags;

    set_tag_value(write_tags, "Result", result);
    if (startpos != "") {
        set_tag_value(write_tags, "FEN", startpos);
    }

    for (std::string tag : {"Event", "Site", "Date", "Round", "White", "Black", "Result"}) {
        if (get_tag_value(write_tags, tag) == "") {
            set_tag_value(write_tags, tag, "?");
        }
    }

    for (auto& [tag_name, tag_value] : write_tags) {
        pgn_ss << "[" << tag_name << " \"" << tag_value << "\"]\n";
    }

    pgn_ss << "\n" << moves_ss.str() << " " << result << "\n";

    return pgn_ss.str();
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


int pgn_parser::write_text_file(std::string path, const std::string &text)
{
    std::ofstream file(path);
    if (file.is_open()) {
        file << text;

        return 0;
    }

    return -1;
}





