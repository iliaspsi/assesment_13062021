-- sum customer basket for all cuisine_parent
-- having at least one order with Breakfast cuisine 
with totalBasket as
(
    select SUM(basket) as userbasket,city 
    from `bi-2019-test.ad_hoc.orders_jan2021` t
    where exists 
        (select * from `bi-2019-test.ad_hoc.orders_jan2021`
         where cuisine_parent = 'Breakfast'
         and user_id = t.user_id
         )
    group by city
)
-- calculate number of breakfast orders per city
-- distinct users that ordered per city (one user may order in different cities)
-- get sum of all user baskets from the cte and divide by distinct users to get average
select 
     count(order_id) as Breakfast_Orders
    ,count(distinct a.user_id) as Breakfast_Users 
    ,a.city
    ,ROUND(MAX(b.userbasket)/ count(distinct a.user_id),2) as Avg_Basket
from `bi-2019-test.ad_hoc.orders_jan2021` a
    inner join totalBasket b 
        on b.city = a.city
where cuisine_parent = 'Breakfast'
group by a.city
having count(order_id)>500
order by 1 desc;




