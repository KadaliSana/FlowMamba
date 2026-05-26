module FlowFeatures;

export {
    redef record Conn::Info += {
        rtt: double &log &optional;
        jitter: double &log &optional;
        packet_rate: double &log &optional;
        mean_iat: double &log &optional;
    };
}

type ConnStats: record {
    start_time: time;
    last_pkt_time: time;
    iat_sum: double;
    iat_count: count;
    last_iat: double;
    iat_diff_sum: double;
    iat_diff_count: count;
    pkt_count: count;
};

global conn_stats_table: table[string] of ConnStats;
global syn_times: table[string] of time;

event connection_SYN_packet(c: connection, pkt: SYN_packet)
    {
    if ( c$uid !in syn_times )
        {
        syn_times[c$uid] = network_time();
        }
    }

event connection_established(c: connection)
    {
    local rtt_val: double;
    if ( ! c?$conn ) return;
    if ( c$uid in syn_times )
        {
        rtt_val = (network_time() - syn_times[c$uid]) / 1sec;
        c$conn$rtt = rtt_val;
        }
    }

event new_packet(c: connection, p: pkt_hdr)
    {
    local uid = c$uid;
    local t = network_time();
    local s: ConnStats;
    local iat: double;
    local diff: double;

    if ( uid !in conn_stats_table )
        {
        s = [
            $start_time=t,
            $last_pkt_time=t,
            $iat_sum=0.0,
            $iat_count=0,
            $last_iat=0.0,
            $iat_diff_sum=0.0,
            $iat_diff_count=0,
            $pkt_count=1
        ];
        conn_stats_table[uid] = s;
        }
    else
        {
        s = conn_stats_table[uid];
        iat = (t - s$last_pkt_time) / 1sec;
        s$iat_sum += iat;
        s$iat_count += 1;
        s$pkt_count += 1;

        if ( s$iat_count > 1 )
            {
            diff = iat - s$last_iat;
            if ( diff < 0.0 )
                {
                diff = -diff;
                }
            s$iat_diff_sum += diff;
            s$iat_diff_count += 1;
            }
        s$last_iat = iat;
        s$last_pkt_time = t;
        }
    }

event connection_state_remove(c: connection) &priority=5
    {
    local uid = c$uid;
    local s: ConnStats;
    local dur: double;

    if ( ! c?$conn ) return;
    if ( uid in conn_stats_table )
        {
        s = conn_stats_table[uid];
        if ( s$iat_count > 0 )
            {
            c$conn$mean_iat = s$iat_sum / s$iat_count;
            }
        else
            {
            c$conn$mean_iat = 0.0;
            }

        if ( s$iat_diff_count > 0 )
            {
            c$conn$jitter = s$iat_diff_sum / s$iat_diff_count;
            }
        else
            {
            c$conn$jitter = 0.0;
            }

        dur = (s$last_pkt_time - s$start_time) / 1sec;
        if ( dur > 0.0 )
            {
            c$conn$packet_rate = s$pkt_count / dur;
            }
        else
            {
            c$conn$packet_rate = 0.0;
            }
        
        delete conn_stats_table[uid];
        }
    else
        {
        c$conn$mean_iat = 0.0;
        c$conn$jitter = 0.0;
        c$conn$packet_rate = 0.0;
        }

    if ( uid in syn_times )
        {
        delete syn_times[uid];
        }
    }
