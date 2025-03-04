#define once




inline int get_target_search_time(int time_left, int time_inc)
{
    int base_time = (time_left / 25) + time_inc;

    return std::min(time_left-100, base_time);
}
