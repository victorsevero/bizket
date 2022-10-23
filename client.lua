HP_UNIT = 0x101

local get_player_hp = function()
    local hp = memory.read_u16_le(0x141924)
    return hp / HP_UNIT
end

local get_player_x = function()
    return memory.read_u16_le(0x13B7F0)
end

local get_player_y = function()
    return memory.read_u16_le(0x13B7F4)
end

local get_boss_hp = function()
    local hp = memory.read_u16_le(0x13BF2C)
    return hp / HP_UNIT
end

local get_boss_x = function()
    return memory.read_u16_le(0x13BEDA)
end

local get_boss_y = function()
    return memory.read_u16_le(0x13BEEE)
end

local make_msg = function(player_hp, player_x, player_y, boss_hp, boss_x, boss_y)
    local msg = ""
    msg = msg .. "ph" .. string.format("%02d", player_hp)
    msg = msg .. " px" .. string.format("%02d", player_x)
    msg = msg .. " py" .. string.format("%02d", player_y)
    msg = msg .. " bh" .. string.format("%02d", boss_hp)
    msg = msg .. " bx" .. string.format("%02d", boss_x)
    msg = msg .. " by" .. string.format("%02d", boss_y)
    return msg
end

while true do
    assert(memory.usememorydomain("MainRAM"))
    local player_hp = get_player_hp()
    local player_x = get_player_x()
    local player_y = get_player_y()
    local boss_hp = get_boss_hp()
    local boss_x = get_boss_x()
    local boss_y = get_boss_y()
    local msg = make_msg(player_hp, player_x, player_y, boss_hp, boss_x, boss_y)
    comm.socketServerSend(msg)
    local response = comm.socketServerResponse()
    emu.frameadvance()
end
