파일명 Caaa_Tb_Pc는는 a도시의 b번째 토픽, c번째 구간을 의미합니다.
이때 c가 0이면 전체기간, c가 1이면 1990~2000, c가 2이면 2000~2010, c가 3이면 2010~입니다. 

우선 포항시에 대하여 네트워크를 추출했습니다.

apriori의 기준은  min support 가 0.01입니다.
association rules의 기준은 confidence가 0.5입니다.
network에 포함시키는 기준은 association rules result들 중에서 lift가 1.5이상인 것들 입니다.
네트워크는 h가 3까지만 깊어지도록 설정했습니다.

해당되는 기사가 많은 토픽/구간도 있고 그렇지 않은 구간도 있어서
네트워크별로 node의 개수가  다릅니다.
특정 네트워크는 노드가 매우 적기도하고 Topic4같은 경우에는 노드가 너무 많아서 시각화가 잘 안되기도 했습니다.
이런 문제를 어떻게 해결할지는 박사님과 함께 논의해보아야 할 것 같습니다.