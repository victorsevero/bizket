PSX_WIDTH = 320
PSX_HEIGHT = 240
BORDER_WIDTH = 84
BORDER_HEIGHT = 0
SCREEN_WIDTH = client.bufferwidth() - 2 * BORDER_WIDTH
SCREEN_HEIGHT = client.bufferheight() - BORDER_HEIGHT


local get_player_hp = function()
    return bit.clear(mainmemory.read_u8(0x141924), 7)
end

local get_player_x = function()
    return mainmemory.read_u16_le(0x1418D2)
end

local get_player_y = function()
    return mainmemory.read_u16_le(0x1418D6)
end

local get_boss_hp = function()
    return bit.clear(mainmemory.read_u8(0x13BF2C), 7)
end

local get_boss_x = function()
    return mainmemory.read_u16_le(0x13BEDA)
end

local get_boss_y = function()
    return mainmemory.read_u16_le(0x13BEDE)
end

local get_camera_x = function()
    return mainmemory.read_u16_le(0x1419BA)
end

local get_camera_y = function()
    return mainmemory.read_u16_le(0x1419BE)
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

local ingame_pixel_to_screen = function(x, y)
    x = SCREEN_WIDTH / PSX_WIDTH * x + BORDER_WIDTH
    x = math.floor(x + 0.5)
    y = SCREEN_HEIGHT / PSX_HEIGHT * y + BORDER_HEIGHT
    y = math.floor(y + 0.5)

    return x, y
end

local draw_player = function()
    local x = get_player_x()
    local y = get_player_y()

    local cam_x = get_camera_x()
    local cam_y = get_camera_y()

    local px_x, px_y = ingame_pixel_to_screen(x - cam_x, y - cam_y)
    gui.drawLine(0, 0, px_x, px_y)
end

local draw_boss = function()
    local x = get_boss_x()
    local y = get_boss_y()

    local cam_x = get_camera_x()
    local cam_y = get_camera_y()

    local px_x, px_y = ingame_pixel_to_screen(x - cam_x, y - cam_y)
    gui.drawLine(0, 0, px_x, px_y)
end

local set_palette = function(palette_idx, color)
    local new_color
    if color == "white" then
        new_color = 0x7FFF
    elseif color == "transparent" then
        new_color = 0x0000
    else
        new_color = 0x8000
    end

    local x_offset = palette_idx % 16
    local y_offset = math.floor(palette_idx / 16)
    local first_address = 0xF0000 + 0x20 * x_offset + 0x800 * y_offset

    for i = 1, 15 do
        local color_address = first_address + i * 2 -- 2 bytes each
        memory.write_u16_le(color_address, new_color, "GPURAM")
    end
end


local set_all_palettes = function()
    -- character
    set_palette(0, "white") -- sprite
    set_palette(1, "white") -- saber
    set_palette(2, "white") -- buster
    set_palette(3, "white") -- mega buster
    set_palette(5, "transparent") -- dash smoke
    set_palette(13, "transparent") -- dash
    set_palette(14, "transparent") -- dash
    set_palette(15, "transparent") -- dash

    -- boss
    set_palette(6, "transparent") -- stomp smoke
    set_palette(24, "transparent") -- projectile hits ground
    set_palette(16, "white") -- damage taken
    set_palette(32, "white")
    set_palette(33, "white")
    set_palette(34, "white")
    set_palette(35, "white") -- projectiles
    set_palette(36, "white")

    -- foreground
    set_palette(103, "transparent")
    set_palette(106, "transparent")

    -- ground
    set_palette(82, "transparent")
    set_palette(83, "transparent")
    set_palette(111, "transparent")
    set_palette(115, "transparent")
    set_palette(116, "transparent")

    --background
    for j = 64, 79 do -- 79
        set_palette(j, "transparent")
    end
    set_palette(119, "transparent")
end


local frameadvance = function()
    -- set_all_palettes()
    emu.frameadvance()
end


local disable_hud = function()
    mainmemory.write_u8(0x1721DF, 0)
end


local palette_search_cycle = function()
    set_palette(i, "transparent")
    print(i - 2)
    if i < 16 * 8 - 1 then
        i = i + 1
    else
        i = 0
    end
end


comm.socketServerSetTimeout(0)
client.invisibleemulation(false)
-- gui.use_surface("client")
-- local i = 0


disable_hud()
frameadvance()
frameadvance()
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
        -- for _ = 1, 10 do
        set_commands(response)
        frameadvance()
        -- end
    end
end
