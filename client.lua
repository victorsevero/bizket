local get_player_hp = function()
    return bit.clear(mainmemory.read_u8(0x141924), 7)
end

local get_boss_hp = function()
    return bit.clear(mainmemory.read_u8(0x13BF2C), 7)

    -- slash beast
    -- return bit.clear(mainmemory.read_u8(0x13BFC8), 7)
end

local make_msg = function(player_hp, boss_hp)
    local msg = ""
    msg = msg .. "ph" .. string.format("%02d", player_hp)
    msg = msg .. " bh" .. string.format("%02d", boss_hp)

    return msg
end

local get_msg = function()
    local player_hp = get_player_hp()
    local boss_hp = get_boss_hp()
    local msg = make_msg(player_hp, boss_hp)

    return msg
end

local set_commands = function(commands)
    local buttons = {}

    buttons["D-Pad Left"] = false
    buttons["D-Pad Right"] = false
    buttons["X"] = false
    buttons["○"] = false
    buttons["□"] = false

    for i = 1, commands:len() do
        local c = commands:sub(i, i)
        if c == "l" then
            buttons["D-Pad Left"] = true
        elseif c == "r" then
            buttons["D-Pad Right"] = true
        elseif c == "x" then
            buttons["X"] = true
        elseif c == "o" then
            buttons["○"] = true
        elseif c == "s" then
            buttons["□"] = true
        end
    end

    joypad.set(buttons, 1)
end

local frameadvance = function()
    emu.frameadvance()
end

local disable_hud = function()
    mainmemory.write_u8(0x1721DF, 0)
end

comm.socketServerSetTimeout(0)

while true do
    local msg = get_msg()
    comm.socketServerSend(msg)
    local response = comm.socketServerScreenShotResponse()
    if response:sub(1, 4) == "load" then
        savestate.loadslot(tonumber(response:sub(-1)))
        -- disable_hud()
        frameadvance()
    elseif response == "close" then
        client.exit()
    elseif response ~= "ok" then
        for _ = 1, 6 do
            set_commands(response)
            frameadvance()
        end
    end
end
