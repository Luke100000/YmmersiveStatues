package net.conczin;

import com.hypixel.hytale.server.core.plugin.JavaPlugin;
import com.hypixel.hytale.server.core.plugin.JavaPluginInit;

import javax.annotation.Nonnull;


public class YmmersiveStatues extends JavaPlugin {
    private static YmmersiveStatues instance;

    public YmmersiveStatues(@Nonnull JavaPluginInit init) {
        super(init);
        instance = this;
    }

    public static YmmersiveStatues getInstance() {
        return instance;
    }
}