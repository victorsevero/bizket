PSX_WIDTH = 320
PSX_HEIGHT = 240
BORDER_WIDTH = 84
BORDER_HEIGHT = 0
SCREEN_WIDTH = client.bufferwidth() - 2 * BORDER_WIDTH
SCREEN_HEIGHT = client.bufferheight() - BORDER_HEIGHT


local get_player_hp = function()
    return bit.clear(mainmemory.read_u8(0x141924), 7)
end

local get_boss_hp = function()
    return bit.clear(mainmemory.read_u8(0x13BF2C), 7)
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
    buttons.Left = false
    buttons.Right = false
    buttons.Cross = false
    buttons.Circle = false
    buttons.Square = false

    for i = 1, commands:len() do
        local c = commands:sub(i, i)
        if c == "l" then
            buttons.Left = true
        elseif c == "r" then
            buttons.Right = true
        elseif c == "x" then
            buttons.Cross = true
        elseif c == "o" then
            buttons.Circle = true
        elseif c == "s" then
            buttons.Square = true
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
disable_hud()
for _ = 1, 60 do
    frameadvance()
end

while true do
    local msg = get_msg()
    comm.socketServerSend(msg)
    local response = comm.socketServerScreenShotResponse()
    if response == "load" then
        savestate.loadslot(3)
        disable_hud()
    elseif response == "close" then
        client.exit()
    elseif response ~= "ok" then
        for _ = 1, 6 do
            set_commands(response)
            frameadvance()
        end
    end
end
