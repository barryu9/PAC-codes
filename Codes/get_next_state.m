function next_state = get_next_state(input_bit,cur_state)
next_state = [input_bit;cur_state(1:end-1)];
end

