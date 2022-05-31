function [gen, det, g] = get_crc_objective(crc_length)
switch crc_length
    case 4
        gen = comm.CRCGenerator('Polynomial', [1, 0, 0, 1, 1], 'InitialConditions', zeros(1, 4), 'FinalXOR', zeros(1, 4));
        det = comm.CRCDetector('Polynomial', [1, 0, 0, 1, 1], 'InitialConditions', zeros(1, 4), 'FinalXOR', zeros(1, 4));
        g = [1, 0, 0, 1, 1];
    case 6
        gen = comm.CRCGenerator('Polynomial', [1, 0, 0, 0, 0, 1, 1], 'InitialConditions', zeros(1, 6), 'FinalXOR', zeros(1, 6));
        det = comm.CRCDetector('Polynomial', [1, 0, 0, 0, 0, 1, 1], 'InitialConditions', zeros(1, 6), 'FinalXOR', zeros(1, 6));
        g = [1, 0, 0, 0, 0, 1, 1];
    case 8
        gen = comm.CRCGenerator('Polynomial', '0xA6', 'InitialConditions', '0x00', 'FinalXOR', '0x00');
        det = comm.CRCDetector('Polynomial', '0xA6', 'InitialConditions', '0x00', 'FinalXOR', '0x00');
        %         g = [1 0 1 0 0 1 1 0 1];
        g = [1, 1, 1, 1, 1, 1, 0, 0, 1];
    case 10
        gen = comm.CRCGenerator('Polynomial', [1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1], 'InitialConditions', zeros(1, 10), 'FinalXOR', zeros(1, 10));
        det = comm.CRCDetector('Polynomial', [1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1], 'InitialConditions', zeros(1, 10), 'FinalXOR', zeros(1, 10));
        g = [1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1];
    case 12
        gen = comm.CRCGenerator('Polynomial', [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1], 'InitialConditions', zeros(1, 12), 'FinalXOR', zeros(1, 12));
        det = comm.CRCDetector('Polynomial', [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1], 'InitialConditions', zeros(1, 12), 'FinalXOR', zeros(1, 12));
        g = [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1];
    case 16
        gen = comm.CRCGenerator('Polynomial', [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1], 'InitialConditions', [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'FinalXOR', [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        det = comm.CRCDetector('Polynomial', [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1], 'InitialConditions', [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'FinalXOR', [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        g = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1];
    case 24
        gen = comm.CRCGenerator('Polynomial', [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1], 'InitialConditions', [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], ...
            'FinalXOR', [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        det = comm.CRCDetector('Polynomial', [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1], 'InitialConditions', [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], ...
            'FinalXOR', [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        g = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1];
    otherwise
        disp('Unsupported CRC length. Program terminates')
end